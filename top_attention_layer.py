

import os
import pickle
import json
import numpy as np
import pandas as pd
import transformers
import torch
import random
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, BertModel, pipeline
from transformers import AdamW, EncoderDecoderModel
from datasets import load_dataset, Dataset
import sys
import argparse
from datetime import datetime

from data import GetValData, TransformersData
from run_files import RunTopAttention
from utils.utils_log import Log                                                 
from utils.utils_retrieve_ood_data import add_ood_datasets

print("\n \n PID is",os.getpid(), "\n \n")


def get_args():

    parser = argparse.ArgumentParser(description="Training model parameters")

    # Arguments for modelling different scenarios
    parser.add_argument(
            "--model_type", type=str, default="bert-base-uncased", 
            help="Model to be used")
    parser.add_argument(
            "--description", type=str, default="model_supervised_attention", 
            help="Description for saved model")
    parser.add_argument(
            "--train_mode", type=str, default="train", 
            help="train to train with attention, or eval for evaluation only")
    parser.add_argument(
            "--random_seed", type=int, default=32, 
            help="Choose the random seed")

    # Arguments for saving/loading models or attention weights
    parser.add_argument(
            "--load_model", type=int, default=0, 
            help="Should we load a saved model or not")
    parser.add_argument(
            "--load_file", type=str, default="filename_to_load.py", 
            help="Model filename to load")
    parser.add_argument(
            "--load_classifier", type=int, default=0, 
            help="Should we load a saved classifier model or not")
    parser.add_argument(
            "--load_file_classifier", type=str, default="filename_to_load.py", 
            help="Model filename to load")

    # Arguments for model training
    parser.add_argument(
            "--epochs", type=int, default=2, 
            help="Number of epochs for training")
    parser.add_argument(
            "--batch_size", type=int, default=8, 
            help="batch_size")
    parser.add_argument(
            "--learning_rate", type=float, default=1e-5, 
            help="Choose learning rate")

    # Arguments for the attention supervision
    parser.add_argument(
            "--lambda_val", type=float, default=1.4, 
            help="Lambda multiplier for explanation loss (between 0 and 1)")
    parser.add_argument(
            "--attention_type", type=str, default="original", 
            help="Which attention weights to use: original, reduced or combined")
    parser.add_argument(
            "--explanation_type", type=str, default='fullexplanations', 
            help="Explanation source: fullexplanations, annotated or combined")
    parser.add_argument(
            "--randomise", type=int, default=0, 
            help="Do we randomise which words we are supervising or not")
    parser.add_argument(
            "--loss", type=str, default='sse', 
            help="sse or kl")

    params, _ = parser.parse_known_args()

    return params


params = get_args()

params.load_model = bool(params.load_model)
params.load_classifier = bool(params.load_classifier)
params.randomise = bool(params.randomise)

# Logging file
log_file_name = 'log_top_layer_attention_' + params.train_mode + "_" \
        + str(os.getpid()) + '.txt'
model_log = Log(log_file_name, params)

# If evaluating only, do this for one epoch
if params.train_mode == 'eval':
    params.epochs = 1

# We set the desription used when we save our model
params.description = params.description  \
        + "___pid_" + str(os.getpid()) \
        + "___batch_size_" + str(params.batch_size) \
        + "___lambda_" + str(params.lambda_val) \
        + "___train_mode_" + str(params.train_mode) \
        + "___random_seed_" + str(params.random_seed) \
        + "___model_type_" + str(params.model_type)

params.description = params.description.replace("/", "")

# Set CUDAS to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if params.model_type[:4] == 'bert':
    model_category = 'bert'

def set_seed(seed_value):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed(params.random_seed)

# Create folder for saving models
if not os.path.exists('savedmodel'):
    os.makedirs('savedmodel')

# Transformers datasets to be loaded
eval_data_list = GetValData(params).eval_data_list

tokenizer = AutoTokenizer.from_pretrained(params.model_type,
                                          truncation=False)

dataset_obj = TransformersData(tokenizer,
        eval_data_list,
        params)

dataset_obj = add_ood_datasets(dataset_obj, tokenizer, params)

# We now load the desired attention weights that we want to attend to
train_data_name = "_esnli_train_"

with open("weights_" + model_category + train_data_name \
        + params.explanation_type \
        +"_all_sentences_saved_" \
        + params.attention_type + ".txt", "rb") as fp:
    weights_all_sentences = pickle.load(fp)

# Format datasets
model = BertModel.from_pretrained(params.model_type)

# If load model is true, we load the NLI model
if params.load_model:
    print("\n \n Loading model \n \n")
    model.load_state_dict(torch.load(params.load_file))


class AttentionClassifier(nn.Module):
    def __init__(self, dimensionality):
        super(AttentionClassifier,self).__init__()

        self.linear1 = torch.nn.Linear(dimensionality, dimensionality)
        self.linear2 = torch.nn.Linear(dimensionality, 1)
        self.tanh = torch.tanh
        self.linear3 = torch.nn.Linear(dimensionality, 3)
        self.sig = torch.nn.Sigmoid()


    def forward(self, x, mask):

        # batch_size x seq_len x dim:
        val = self.linear1(x)
        val = self.tanh(val)
        
        # batch_size x seq_len x 1:
        val = self.linear2(val)
        val = self.sig(val)
        
        # Use mask so we do not pay attention to padding tokens
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        
        val = torch.einsum('ijk, ijk -> ijk', [val, mask])
        
        # batch_size
        sum_val = torch.einsum('ijk -> i', val)
        
        # batch_size
        inv_sum_val = 1/sum_val
        
        # batch_size x seq_len x 1:
        att_weights = torch.einsum('ijk, i -> ijk', [val, inv_sum_val])
        
        #batch_size x dimensions
        updated_representation = torch.einsum(
                'ijk, ijm -> ik', [x, att_weights])
        
        output_val = self.linear3(updated_representation)
        
        return output_val, att_weights


attention_and_classifier = AttentionClassifier(768)

if params.load_classifier:
    print("\n \n Loading attention \n \n")
    attention_and_classifier.load_state_dict(
            torch.load(params.load_file_classifier))

attention_and_classifier.to(device)
model.to(device)

# Create optimizer
optimizer = AdamW(list(model.parameters()) + list(
    attention_and_classifier.parameters()), lr=params.learning_rate)

run = RunTopAttention(
        model, 
        attention_and_classifier, 
        dataset_obj, 
        optimizer, 
        params, 
        model_log,
        weights_all_sentences)

# Train model
for epoch in range(params.epochs):

    if params.train_mode != "eval":
        run.train_model()

        now = datetime.now()
        model_log.msg([str(now.strftime("%d/%m/%Y %H:%M:%S")), 
            str(epoch+1) + " epochs have passed ----------"])

        torch.save(run.model.state_dict(),
                os.getcwd() + "/savedmodel/" + "_description_" \
                        + params.description + "_epoch_" + str(epoch) \
                        + "_saved_dict.pt")

        torch.save(attention_and_classifier.state_dict(), 
                os.getcwd() + "/savedmodel/" + "att_and_class__description_" \
                + params.description + "_epoch_" + str(epoch)  \
                + "_saved_dict.pt")

    for dataset_name, dataset in dataset_obj.eval_returned_dataloaders_dict.items():
        run.evaluate_model(dataset, dataset_name)



