

import os
import sklearn.metrics
import random
import json
import numpy as np
import pandas as pd
import transformers
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from datasets import load_dataset, Dataset
import sys
import argparse
from datetime import datetime
import scipy.stats
import random
import math

from data import GetValData, TransformersData
from utils.utils_log import Log

class RunAttention():
    """
    Training and evaluation for model with supervised attention

    Attributes:
        model: huggingface model, bert-based-uncased or microsoft/deberta-large
        dataset_obj: contains train dataloader
        params: trainng params, including lambda val and which heads supervised
        log: log to record ID and OOD performance
        optimizer: optimizer
        weights_all_sentences: desired attention weights (pre normalization)
        device: device for moving tensors to gpu
        model_category: either bert or deberta model

    """
    def __init__(
            self, 
            model, 
            dataset_obj: TransformersData, 
            optimizer: transformers.optimization.AdamW, 
            params_training: argparse.Namespace, 
            model_log: Log, 
            attention_weights: dict) -> None:

        self.model = model
        self.dataset_obj = dataset_obj
        self.params = params_training
        self.log = model_log
        self.optimizer = optimizer
        self.weights_all_sentences = attention_weights

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Which type of model are we using
        if self.params.model_type[:4] == 'bert':
            self.model_category = 'bert'
        elif self.params.model_type[:9] == 'microsoft':
            self.model_category = 'deberta'


    def predictions_class_update(
            self, 
            predictions: torch.tensor, 
            dataset_description: str) -> torch.tensor:
        """
        Combine neutral and contradiction predictions if 2 class dataset

        Args:
            predictions: class predictions for minibatch
            dataset_description: name of the dataset being evaluated

        Returns:
            predictions: two class predictions (0 or 1)

        """
        two_class_datasets = ['hans']

        if dataset_description in two_class_datasets:
            
            for i, j in enumerate(predictions):
                if j == 2:
                    predictions[i] = 1

        return predictions


    @torch.no_grad()
    def evaluate_model(
            self, 
            dataset: torch.utils.data.dataloader.DataLoader, 
            description: str) -> None:
        """
        Evaluates dataset and prints accuracy

        Args:
            dataset: dataloader
            description: description of the dataset

        Returns:
            None
        """
        self.model.eval()

        # For calculating accuracy
        correct_observations = 0
        obs_trained_on = 0

        # Forward pass for validation data    
        for i, batch_eval in enumerate(dataset):
            batch_eval = {k: v.to(self.device) \
                    for k, v in batch_eval.items()}
            
            outputs = self.model(
                    batch_eval['input_ids'], 
                    batch_eval['attention_mask'], 
                    return_dict=True)
            
            # Shape of batch and seq length
            size_in_batch = batch_eval['input_ids'].shape[0]
            seq_len = batch_eval['input_ids'].shape[1]

            attention_output = \
                    outputs['attentions'][self.params.attention_layer]

            # Find predictions
            value, prediction = outputs['logits'].max(1)
            prediction = self.predictions_class_update(prediction, description)
            
            # Calculating accuracy
            correct_obs = \
                    int((batch_eval['label'] == prediction).float().sum())
            correct_observations = correct_observations + correct_obs
            
            obs_trained_on = obs_trained_on + size_in_batch

        # Output statistics
        self.log.msg(["Printing evaluation for " + description])
        self.log.msg([str(correct_observations) + " correct observations",
        str(obs_trained_on) + " total observations",
        str(round(correct_observations/obs_trained_on,4)) +  " accuracy"])


    def train_model(self) -> None:
        """
        Training on the NLI task and with auxiliary attention loss

        Returns:
            None
        """
        self.model.train()

        dataloader = self.dataset_obj.train_dataloader_dict['training_data']

        obs_trained_on = 0
        correct_obs = 0

        for i, j in enumerate(dataloader):
         
             print("Batch number:", i)
             batch = j
             batch = {k: v.to(self.device) for k, v in batch.items()}
 
             # Shape of batch and seq length
             size_in_batch = batch['input_ids'].shape[0]
             seq_len = batch['input_ids'].shape[1]

             # We find the model outputs and the attention values

             # Note, we could also pass token_type_ids but we find this
             # ..doesn't impact ID \ OOD performance, or atttention 
             # ..behaviour when training over the entire SNLI dataset
             outputs = self.model(
                     batch['input_ids'], 
                     batch['attention_mask'], 
                     return_dict=True)
             attention_output = \
                     outputs['attentions'][self.params.attention_layer]
             
             # Find attention weights for batch
             desired_att_weights_0s_1s = self.batch_att_weights(
                     size_in_batch, 
                     seq_len, 
                     batch['input_ids'], 
                     self.weights_all_sentences, 
                     batch['attention_mask'])

             correct = int(batch['label'].eq(torch.max(
                 outputs['logits'],1)[1]).sum())

             correct_obs = correct_obs + correct

             # We calculate the NLI and attention loss
             loss_nli = F.cross_entropy(
                     outputs['logits'], 
                     batch['label'])

             attention_loss = \
                     self.calculate_attention_loss_original(
                             desired_att_weights_0s_1s, 
                             attention_output, 
                             size_in_batch, 
                             seq_len, 
                             batch['attention_mask'])
 
             # Find combined loss
             combined_loss = self.params.lambda_val * torch.true_divide(
                     attention_loss,
                     float(self.params.attention_heads)) + loss_nli

             combined_loss.backward()

             self.optimizer.step()
             
             self.optimizer.zero_grad()
             
             obs_trained_on = obs_trained_on + size_in_batch

        self.log.msg(["Training accuracy " \
                + str(round((correct_obs / obs_trained_on), 4))])
        

    def batch_att_weights(
            self, 
            size_in_batch: int, 
            seq_len: int, 
            token_ids: torch.tensor, 
            weights_all_sentences: dict, 
            attention_mask: torch.tensor) -> torch.tensor:

        """
        Create a tensor of desired attention weights for the batch

        Args:
            size_in_batch: number of observations in the minibatch
            seq_len: token length for each example (including padding)
            token_ids: input_ids for minibatch
            weights_all_sentences: each input id (minus padding) has a
                ... corresponding list of 1s and 0s as desired attention
                ... with the list matching the input id length (no padding)
            attention_mask: attention mask for the batch

        Returns:
            desired_att_weights_0s_1s: desired attention values for the batch
        """
        
        desired_att_weights_0s_1s = torch.zeros(size_in_batch, seq_len)
        
        # Creating tensor of desired attention weights
        for row_no in range(size_in_batch):

            input_ids = list(np.array(token_ids[row_no].cpu()))
            input_ids = [input_ids[idx] for idx in range(len(input_ids)) \
                    if attention_mask[row_no, idx] != 0]

            token_len = len(weights_all_sentences[str(input_ids)])

            if self.params.randomise:
                weights_all_sents = self.randomise(
                        weights_all_sentences[str(input_ids)], 
                        token_ids[row_no])
            else:
                weights_all_sents = weights_all_sentences[str(input_ids)]

            desired_att_weights_0s_1s[row_no,:token_len] = \
                    torch.tensor(weights_all_sents)
        
        # Note: desired_att_weights_0s_1s may also contain 0.5 values on the
        # .. 'combined' attention weights setting
        # There may also be a -1 in the first position, indicating not to
        # .. supervise the model using this explanation

        # Tensor to GPU
        desired_att_weights_0s_1s = desired_att_weights_0s_1s.to(self.device)

        return desired_att_weights_0s_1s


    def randomise(
            self, 
            weights: np.array, 
            token_ids: torch.tensor) -> np.array:
        """
        We randomise which words we should be paying attention to ...
        .. randomising the order of the weights separately in each sentence

        Args:
            weights: desired attention weights for a premise / hypothesis pair
            token_ids: token ids for the example

        Returns:
            weights: weights after separately shuffling hypothesis and premise
        """
        if self.model_category == 'bert':
            cls = 101
            sep = 102

        elif self.model_category == 'deberta':
            cls = 1
            sep = 2

        sep1_index = (token_ids == sep).int().nonzero().min().flatten().item()
        end_token = (token_ids == sep).int().nonzero().max().flatten().item()

        weights = np.array(weights)

        # Models can have multiple sep tokens inbetween the sentences
        if self.model_category == 'bert':
            hyp_start = sep1_index + 1
        elif self.model_category == 'deberta':
            hyp_start = sep1_index + 1

        premise_indices = np.arange(1, sep1_index)
        hyp_indices = np.arange(hyp_start, end_token)

        premise_array = weights[premise_indices]
        hyp_array = weights[hyp_indices]

        # We now shuffle each array
        random.shuffle(premise_array)
        random.shuffle(hyp_array)

        weights[premise_indices] = premise_array
        weights[hyp_indices] = hyp_array

        return weights


    def calculate_attention_loss_original(
            self, 
            desired_att_weights: torch.tensor,
            attention_output: torch.tensor, 
            size_in_batch: int, 
            seq_len: int, 
            att_mask: torch.tensor) -> torch.tensor:
        """
        We find the difference between the current and the desired attention

        Args:
            desired_att_weights: desired attention values for the batch (0s,1s)
            attention_output: attention output for layer being supervised
            size_in_batch: observations in minibatch
            seq_len: sequence length of input_ids (including padding)
            att_mask: attention masks for minibatch

        Returns:
            all_attention_loss: attention loss for the minibatch

        """
        # Create loss tensor
        all_attention_loss = torch.tensor(0).to(self.device)

        # First we calculate the attention weight that we need
        for i_val in range(size_in_batch):
            
            attention_mask = att_mask[i_val]
            # We scale the desired weight tensor to 1
            weight_mult = torch.sum(desired_att_weights[i_val, :])
        
            # We only scale the weights if the first value is not -1
            if desired_att_weights[i_val,:][0] >= 0:
                
                desired_att_weights[i_val, :] = \
                        desired_att_weights[i_val, :] / weight_mult
                
                assert torch.allclose(
                    torch.sum(desired_att_weights[i_val, :]), 
                    torch.tensor([1.0]).to(self.device))

            if self.params.loss == 'sse':

                # We loop over every head
                for head_no in self.params.attention_head_numbers:
            
                    single_head_CLS_att = attention_output[i_val,head_no,0,:]
                    # Calculate loss for specific head
                    loss_atten_head = single_head_CLS_att \
                            - desired_att_weights[i_val, :]
                
                    # Include loss if first element not -1
                    if desired_att_weights[i_val,:][0] >= 0:

                        all_attention_loss = all_attention_loss \
                                + (torch.sum(torch.square(loss_atten_head)))
                 
            elif self.params.loss == 'kl':

                single_head_CLS_att = torch.mean(
                        attention_output[i_val,:,0,:],
                        0)

                epsilon = torch.tensor([0.0001]).to(self.device)
                
                a = desired_att_weights[i_val, :]
                a = torch.where(a == 0, epsilon, a)
                a = torch.true_divide(a,torch.sum(a))
                b = single_head_CLS_att
                b = torch.true_divide(b, torch.sum(b))

                # Find the KL divergence:
                kld = torch.tensor([0]).to(self.device)
                
                if 0 in attention_mask:
                    max_len = (attention_mask == 0).int().nonzero().min()
                else:
                    max_len = attention_mask.shape[0]
                    
                for i in range(max_len):
                    if a[i] != torch.tensor([0.0]).to(self.device):
                        kld = kld + (-1)* a[i] * torch.log(b[i]/a[i]).to(
                                self.device)
                all_attention_loss = all_attention_loss + kld

        # This counteracts dividing by the number of attention heads
        if self.params.loss == 'kl':
            all_attention_loss = all_attention_loss \
                    * float(self.params.attention_heads)

        return all_attention_loss


class RunTopAttention():
    """
    Training and evaluation for model with additional attention layer

    Attributes:
        model: huggingface model, using bert-based-uncased
        atten_and_classifier: additional attention layer to be supervised
        dataset_obj: contains train dataloader
        params: training params, containing lambda value
        log: log to record ID and OOD performance
        optimizer: optimizer
        weights_all_sentences: desired attention weights (pre normalization)
        device: device for moving tensors to GPU
    """
    def __init__(
            self, 
            model: transformers.models.bert.modeling_bert.BertModel, 
            atten_and_classifier,
            dataset_obj: TransformersData, 
            optimizer: transformers.optimization.AdamW, 
            params_training: argparse.Namespace, 
            model_log: Log, 
            attention_weights: dict) -> None:

        self.model = model
        self.dataset_obj = dataset_obj
        self.params = params_training
        self.log = model_log
        self.optimizer = optimizer
        self.weights_all_sentences = attention_weights
        self.attention_and_classifier = atten_and_classifier

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def predictions_class_update(
            self, 
            predictions: torch.tensor, 
            dataset_description: str) -> torch.tensor:
        """
        Combine neutral and contradiction predictions if 2 class dataset

        Args:
            predictions: class predictions for minibatch
            dataset_description: name of the dataset being evaluated

        Returns:
            predictions: two class predictions (0 or 1)

        """
        two_class_datasets = ['hans']
        
        if dataset_description in two_class_datasets:

            for i, j in enumerate(predictions):
                if j == 2:
                    predictions[i] = 1

        return predictions


    @torch.no_grad()
    def evaluate_model(
            self, 
            dataset: torch.utils.data.dataloader.DataLoader, 
            description: str) -> None:
        """
        Evaluates dataset and prints accuracy
    
        Args:
            dataset: dataloader
            description: description of the dataset

        Returns:
            None
        """
        
        self.attention_and_classifier.eval()
        self.model.eval()

        # For calculating accuracy
        correct_observations = 0
        total_observations = 0
        
        for i, batch_eval in enumerate(dataset):

            batch_eval = {k: v.to(self.device) \
                    for k, v in batch_eval.items()}
            
            # Shape of batch and seq length
            size_in_batch = batch_eval['input_ids'].shape[0]
            seq_len = batch_eval['input_ids'].shape[1]

            bert_outputs = self.model(
                    batch_eval['input_ids'], 
                    batch_eval['attention_mask'])

            bert_seq_output = bert_outputs['last_hidden_state']
           
            model_outputs, attention_output = self.attention_and_classifier(
                    bert_seq_output, 
                    batch_eval['attention_mask'])

            value, prediction = model_outputs.max(1)

            prediction = self.predictions_class_update(prediction, description)

            correct_obs = \
                    int((batch_eval['label'] == prediction).float().sum())
            correct_observations = correct_observations + correct_obs

            total_observations = total_observations + model_outputs.shape[0]

        self.log.msg(["Printing evaluation for " + description])
        self.log.msg([str(correct_observations) + " correct observations",
        str(total_observations) + " total observations",
        str(round(correct_observations/total_observations,4)) +  " accuracy"])


    def train_model(self) -> None:
        """
        Training on the NLI task and with auxiliary attention loss

        Returns:
            None
        """
        self.model.train()
        self.attention_and_classifier.train()
        
        dataloader = self.dataset_obj.train_dataloader_dict['training_data']

        obs_trained_on = 0
        correct_obs = 0

        for i, j in enumerate(dataloader):
             
             print("Batch number:", i)
             batch = j
             batch = {k: v.to(self.device) for k, v in batch.items()}

             # Shape of batch and seq length
             size_in_batch = batch['input_ids'].shape[0]
             seq_len = batch['input_ids'].shape[1]

             # We find the NLI loss
             bert_outputs = self.model(
                     batch['input_ids'], 
                     batch['attention_mask'], 
                     return_dict=True)

             bert_seq_output = bert_outputs['last_hidden_state']
             model_outputs, attention_output = self.attention_and_classifier(
                     bert_seq_output, 
                     batch['attention_mask'])
             
             # Find attention weights for batch and create attention mask
             desired_att_weights_0s_1s = self.batch_att_weights(
                     size_in_batch, 
                     seq_len, 
                     batch['input_ids'])

             correct = int(
                     batch['label'].eq(torch.max(model_outputs,1)[1]).sum())
             correct_obs = correct_obs + correct
            
             # We calculate the NLI and attention loss
             loss_nli = F.cross_entropy(model_outputs, batch['label'])

             attention_loss = \
                     self.calculate_attention_loss_original(
                             desired_att_weights_0s_1s,
                             attention_output, 
                             size_in_batch, 
                             seq_len)
             
             # Find combined loss
             combined_loss = self.params.lambda_val * (attention_loss) \
                     + loss_nli

             combined_loss.backward()
             self.optimizer.step()
             self.optimizer.zero_grad()
             
             obs_trained_on = obs_trained_on + size_in_batch

        self.log.msg(["Training accuracy " \
                + str(round((correct_obs / obs_trained_on), 4))])


    def batch_att_weights(
            self, 
            size_in_batch: int, 
            seq_len: int, 
            token_ids: torch.tensor) -> torch.tensor:
        """
        We create a tensor of desired attention weights

        Args:
            size_in_batch: number of observations in minibatch
            seq_len: token length for each example (including padding)
            token_ids: input ids for minibatch

        Returns:
            desired_att_weights_0s_1s: desired attention values for the batch
        """

        # Create tensor of desired attention weights
        desired_att_weights_0s_1s = torch.zeros(size_in_batch, seq_len)
        
        for row_no in range(size_in_batch):

            input_ids = list(np.array(token_ids[row_no].cpu()))
            input_ids = [x for x in input_ids if x != 0]
            token_len = len(self.weights_all_sentences[str(input_ids)])
 
            desired_att_weights_0s_1s[row_no,:token_len] = torch.tensor(
                    self.weights_all_sentences[str(input_ids)])
        
        return desired_att_weights_0s_1s


    def calculate_attention_loss_original(
            self, 
            desired_att_weights: torch.tensor,
            attention_output: torch.tensor, 
            size_in_batch: int, 
            seq_len: int) -> torch.tensor:

        """
        We compare current attention to desired attention

        Args:
            desired_att_weights: 1s and 0s for each observation in minibatch
            attention_output: attention output for minibatch
            size_in_batch: number of observations in minibatch
            seq_len: sequence length
        
        Returns:
            all_attention_loss: attention loss
        """

        # Move tensors to device
        current_attention = attention_output
        desired_att_weights = desired_att_weights.to(self.device)

        # Create tensor of desired weights
        desired_att_weights = self.create_desired_weights(
                desired_att_weights, 
                current_attention, 
                size_in_batch)  

        # Calculate difference between current and desired attention weights
        current_attention = current_attention.flatten(1)
        att_diff = current_attention - desired_att_weights
    
        # If first attention value is -1, we do not supervise
        for i_val in range(size_in_batch):
            if desired_att_weights[i_val,:][0] == -1:
                att_diff[i_val,:] = current_attention[i_val,:] \
                        - current_attention[i_val,:]

        # We calculate the sum of squared errors for the attention loss
        all_attention_loss = torch.sum((torch.square(att_diff)))
        
        return all_attention_loss


    def create_desired_weights(
            self, 
            desired_att_weights: torch.tensor, 
            current_attention: torch.tensor, 
            size_in_batch: int) -> torch.tensor:
        """
        We scale the desired attention weights

        Args:
            desired_att_weights: 1s and 0s for desired attention weignts
            current_attention: model attention output
            size_in_batch: observations in minibatch

        Returns:
            desired_att_weights: weights afer scaling so they sum to 1
        """
        # For each observation we scale the suggested attention weights
        for i_val in range(size_in_batch):

            # If the first value is -1, we do not supervise the attention
            if desired_att_weights[i_val,:][0] != -1:

                # We calculate scaling factor
                initial_att_batch = sum(current_attention[i_val,:])
                expected_att = sum(desired_att_weights[i_val,:])
                weight_mult = float(initial_att_batch / expected_att)

                # Scale weights
                desired_att_weights[i_val,:] = \
                        desired_att_weights[i_val,:]*weight_mult

        return desired_att_weights
