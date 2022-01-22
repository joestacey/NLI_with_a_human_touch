

import os
import csv
import json
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
import nltk
import argparse


def get_args():

  parser = argparse.ArgumentParser(description="Training model parameters")

  parser.add_argument(
          "--model_type", type=str, default="bert-base-uncased", 
          help="Model to be used")
  parser.add_argument(
          "--batch_size", type=int, default=8, 
          help="batch_size")
  parser.add_argument(
          "--random_seed", type=int, default=42, 
          help="Choose random seed")
  parser.add_argument(
          "--explanation_type", type=str, default="fullexplanations", 
          help="Which explanations to use")
  params, _ = parser.parse_known_args()

  return params


class GetValData():

    """
    Creates a dictionary with specifications for the validation data
    """
    def __init__(self, params, snli=True, mnli=True, anli=True):

        # Which datasets to load
        self.snli_to_load = snli
        self.mnli_to_load = mnli
        self.anli_to_load = anli
        
        # To contain dictionaries for each of the required validation sets
        self.eval_data_list = []

        self.run()


    def run(self):
        """
        We append dictionaries for each validation dataset we want
        """
        if self.snli_to_load:
            self.load_snli()

        if self.mnli_to_load:
            self.load_mnli()

        if self.anli_to_load:
            self.load_anli()


    def load_snli(self):
        """
        Appending SNLI dictionary
        """
        for split in ['test', 'validation']:
            self.eval_data_list.append(
                    {'description': 'snli', 
                        'split_name': split,
                        'premise_name': 'premise', 
                        'hypothesis_name': 'hypothesis'})


    def load_mnli(self):
        """
        Appending MNLI dictionary
        """
        for split in ['validation_mismatched', 'validation_matched']:
            self.eval_data_list.append(
                    {'description': 'multi_nli', 
                        'split_name': split,
                        'premise_name': 'premise', 
                        'hypothesis_name': 'hypothesis'})


    def load_anli(self):
        """
        Appending ANLI dictionaries
        """
        for split in ['dev_r1', 'dev_r2', 'dev_r3', 'test_r1', 
                      'test_r2', 'test_r3']:
            self.eval_data_list.append(
                    {'description': 'anli', 
                        'split_name': split, 
                        'premise_name': 'premise', 
                        'hypothesis_name': 'hypothesis'})


class TransformersData():

    """
    We load and tokenize our huggingface training and validation data

    """
    def __init__(self, tokenizer_, eval_data_list_,  params):

        """
        Sets attributes and run self.run() method
        """
        # Parameters
        self.params = params
        
        # Set the columns required for our tokenized data
        self.model_type = params.model_type
        self.data_cols = columns=['input_ids', 'token_type_ids', 
                                  'attention_mask', 'label']
        
        # Dictionaries for huggingface datasets to be loaded (train and eval)
        self.eval_dataset_list = eval_data_list_

        # Tokenized datasets (used to create attention weights)
        self.eval_returned_datasets_dict = {}
        self.train_returned_dataset_dict = {}
        # For eSNLI dataset with explanations:
        self.dev_returned_dataset_dict = {}
        self.test_returned_dataset_dict = {}

        # For the eSNLI test and train data with explanations
        self.loaded_train_data = None
        self.loaded_dev_data = None
        self.loaded_test_data = None

        # Other parameters
        self.seed_no = params.random_seed
        self.batch_size = params.batch_size
        self.eval_returned_dataloaders_dict = {}
        
        # Dataloaders:
        # Train dataloader used directly in training / evaluation
        self.train_dataloader_dict = {}
        self.dev_dataloader_dict = {}
        self.test_dataloader_dict = {}

        self.tokenizer = tokenizer_
 
        # Loading eSNLI data
        self.esnli = pd.DataFrame()
        dataset1 = pd.read_csv("dataset_esnli/esnli_train_1.csv")
        dataset2 = pd.read_csv("dataset_esnli/esnli_train_2.csv")
        self.esnli = pd.concat([dataset1, dataset2])
        self.esnli_test = pd.read_csv("dataset_esnli/esnli_test.csv")
        self.esnli_dev = pd.read_csv("dataset_esnli/esnli_dev.csv")

        # Run method
        self.run()


    def tokenize_data(self, loaded_data, dataset_dict):
        """
        We tokenize the data
        """
        # Information to store with dataset
 
        loaded_data = loaded_data.map(
                lambda x: self.tokenizer(x[dataset_dict['premise_name']],
                    x[dataset_dict['hypothesis_name']], 
                    truncation=False, 
                    padding=True), 
                batched=True, batch_size=self.batch_size)

        return loaded_data


    def get_all_eval_data(self):
        """
        Gets loaded datasets for all datasets to be used in evaluation
        """

        for dataset_dict in self.eval_dataset_list:
            
            # Load dataset
            loaded_data = load_dataset(
                    dataset_dict['description'], 
                    split=dataset_dict['split_name'])

            # Remove examples with no gold label
            loaded_data = loaded_data.filter(
                    lambda example: example['label'] in [0, 1, 2])

            print("\n\nDataset:", dataset_dict['description'])

            loaded_data = self.tokenize_data(loaded_data, dataset_dict)

            self.eval_returned_datasets_dict.update(
                    {dataset_dict['description'] + "_" \
                            + dataset_dict['split_name'] : loaded_data})

        
    def format_esnli(self, df, train=False):

        # Updating column names
        df['premise'] = df['Sentence1']
        df['hypothesis'] = df['Sentence2']
        df['explanation_1'] = df['Explanation_1']

        if not train:
            df['explanation_2'] = df['Explanation_2']
            df['explanation_3'] = df['Explanation_3']

            if self.params.explanation_type == 'annotated' or \
                    self.params.explanation_type == 'annotated_stopwords':
                
                df['explanation_2'] = df['Sentence1_marked_2'] + ' [SEP] ' \
                        + df['Sentence2_marked_2']
                df['explanation_3'] = df['Sentence1_marked_3'] + ' [SEP] ' \
                        + df['Sentence2_marked_3']

        # For test or train
        if self.params.explanation_type == 'annotated' \
                or self.params.explanation_type == 'annotated_stopwords':
            df['explanation_1'] = df['Sentence1_marked_1'] + ' [SEP] ' \
                    + df['Sentence2_marked_1']

        df['label'] = df['gold_label']

        # Label needs to be 0, 1 or 2
        df['label'].loc[df['label'] == 'entailment'] = 0
        df['label'].loc[df['label'] == 'neutral'] = 1
        df['label'].loc[df['label'] == 'contradiction'] = 2


        df['premise'] = df['premise'].apply(lambda x: str(x))
        df['hypothesis'] = df['hypothesis'].apply(lambda x: str(x))

        if not train:
            df = df[['premise', 'hypothesis', 'explanation_1', 
                'explanation_2', 'explanation_3', 'label']]
        else:
            df = df[['premise', 'hypothesis', 'explanation_1', 'label']]

        return df


    def get_esnli_data(self, data_type):
        """
        Gets dataset to be used for training (shuffling the dataset)
        This may also contain the explanation
        """

        if data_type == 'train':
            df = self.esnli

        elif data_type == 'dev':
            df = self.esnli_dev

        elif data_type == 'test':
            df = self.esnli_test

        is_train = (data_type == 'train')

        df = self.format_esnli(df, is_train)
        
        # If we're using annotations instead of the free text explanations
        loaded_data = Dataset.from_pandas(df)

        # Remove examples with no gold label
        loaded_data = loaded_data.filter(
                lambda example: example['label'] in [0, 1, 2])

        if data_type == 'train':
            loaded_data = loaded_data.shuffle(seed=self.seed_no)
            self.loaded_train_data = loaded_data

        elif data_type == 'dev':
            self.loaded_dev_data = loaded_data

        elif data_type == 'test':
            self.loaded_test_data = loaded_data

        if data_type == 'train':

            data_dict = {'description': 'esnli', 
                    'explanation_name': 'explanation_1',
                    'split_name': 'train', 
                    'premise_name': 'premise', 
                    'hypothesis_name': 'hypothesis'}
        
        elif data_type == 'dev':

             data_dict = {'description': 'esnli_csv', 
                     'explanation_name': 'explanation_1',
                     'explanation2_name': 'explanation_2', 
                     'explanation3_name': 'explanation_3',
                     'split_name': 'validation', 
                     'premise_name': 'premise', 
                     'hypothesis_name': 'hypothesis'}

        elif data_type == 'test':

            data_dict = {'description': 'esnli_csv', 
                    'explanation_name': 'explanation_1',
                    'explanation2_name': 'explanation_2', 
                    'explanation3_name': 'explanation_3',
                    'split_name': 'test', 
                    'premise_name': 'premise', 
                    'hypothesis_name': 'hypothesis'}

        loaded_data = self.tokenize_data(loaded_data, data_dict)

        if data_type == 'train':

            self.train_returned_dataset_dict.update({'data':  loaded_data})

        elif data_type == 'test':

            self.test_returned_dataset_dict.update({'data':  loaded_data})

        elif data_type == 'dev':

             self.dev_returned_dataset_dict.update({'data':  loaded_data})


    def prepare_dataloaders(self, loaded_dataset, dataset_name, data_type):
        """
        Creates a dataloader for a loaded dataset
        """

        loaded_dataset.set_format(type='torch', columns=self.data_cols)
        dataloader = torch.utils.data.DataLoader(
                loaded_dataset, 
                batch_size=self.batch_size)

        if data_type == 'train':

            self.train_dataloader_dict.update(
                    {dataset_name: dataloader})

        elif data_type == 'eval':

            self.eval_returned_dataloaders_dict.update(
                    {dataset_name: dataloader})

        elif data_type == 'test':

            self.test_dataloader_dict.update(
                    {dataset_name: dataloader})

        elif data_type == 'dev':

            self.dev_dataloader_dict.update(
                    {dataset_name: dataloader})


    def prepare_all_eval_dataloaders(self):
        """
        Prepares dataloaders for all dataloaders to be used in evaluation
        """

        for dataset_name, loaded_dataset in self.eval_returned_datasets_dict.items():

            self.prepare_dataloaders(loaded_dataset, dataset_name, 'eval')


    def prepare_train_dataloader(self):
        """
        Prepares a dataloader for the training data
        """
        self.prepare_dataloaders(
                self.train_returned_dataset_dict['data'], 
                'training_data', 
                'train')


    def prepare_test_dataloader(self):
        """
        Prepares a dataloader for the eSNLI test data
        """

        self.prepare_dataloaders(
                self.test_returned_dataset_dict['data'], 
                'test_data', 
                'test')


    def prepare_dev_dataloader(self):
        """
        Prepares a dataloader for the eSNLI dev data
        """

        self.prepare_dataloaders(
                self.dev_returned_dataset_dict['data'], 
                'dev_data', 
                'dev')


    def run(self):
        """
        Run method, creates dataloaders for evaluation and train dataloader
        """
        self.get_all_eval_data()
        self.prepare_all_eval_dataloaders()
        self.get_esnli_data(data_type='train') # For eSNLI training data
        self.get_esnli_data(data_type='dev') # For eSNLI dev data
        self.get_esnli_data(data_type='test') # For eSNLI test data
        self.prepare_train_dataloader()
        self.prepare_test_dataloader()
        self.prepare_dev_dataloader()


class AddSnliHard():
    """
    We load and tokenize the SNLI-Hard dataset
    """
    def __init__(self, dataset_obj, description, filename, params, tokenizer):

        self.output_data_dict = {}

        batch_size = params.batch_size

        snli_hard_df = pd.DataFrame(columns=['s1', 's2', 'labels'])
        
        with open(filename) as f:
            data = list(f)

        for json_str in data:
            result = json.loads(json_str)
            snli_hard_df = snli_hard_df.append(
                    {'s1': result['sentence1'], 
                    's2': result['sentence2'], 
                    'labels': result['gold_label']}, ignore_index=True)

        snli_hard_df['label'] = -1
        snli_hard_df['label'].loc[snli_hard_df['labels'] == 'entailment'] = 0
        snli_hard_df['label'].loc[snli_hard_df['labels'] == 'neutral'] = 1
        snli_hard_df['label'].loc[snli_hard_df['labels'] == 'contradiction'] = 2

        snli_hard = Dataset.from_pandas(snli_hard_df)

        snli_hard = snli_hard.filter(
                lambda example: example['label'] in [0, 1, 2])

        snli_hard = snli_hard.map(lambda x: tokenizer(x['s1'], x['s2'],
                                                    truncation=False,
                                                    padding=True), 
                                        batched=True, batch_size=batch_size)

        snli_hard.set_format(
                type='torch', 
                columns=['input_ids', 'token_type_ids', 
                    'attention_mask', 'label'])

        dataloader_snli_hard = torch.utils.data.DataLoader(
                snli_hard, 
                batch_size=batch_size)

        self.output_data_dict.update({description: dataloader_snli_hard})


class AddHANS():
    """
    We load and tokenize the SNLI-Hard dataset
    """
    def __init__(
            self, 
            dataset_obj, 
            description, 
            filename, 
            params, 
            tokenizer, 
            heuristic_list):

        self.output_data_dict = {}

        batch_size = params.batch_size
        
        fi = open(filename, "r")

        correct_dict = {}
        first = True

        for line in fi:
            if first:
                labels = line.strip().split("\t")
                idIndex = labels.index("pairID")
                first = False
                continue
            else:
                parts = line.strip().split("\t")
                this_line_dict = {}
                for index, label in enumerate(labels):
                    if label == "pairID":
                        continue
                    else:
                        this_line_dict[label] = parts[index]
                correct_dict[parts[idIndex]] = this_line_dict

        hans_df = pd.DataFrame.from_dict(correct_dict, orient='index')

        hans_df = hans_df[hans_df['heuristic'].isin(heuristic_list)]

        hans_df = hans_df[['sentence1', 'sentence2', 'gold_label']]
        hans_df = hans_df.rename(
                columns={'sentence1': 's1', 
                        'sentence2': 's2', 
                        'gold_label': 'label'})

        hans_df['label'].loc[hans_df['label'] == 'entailment'] = 0
        hans_df['label'].loc[hans_df['label'] == 'non-entailment'] = 1

        hans = Dataset.from_pandas(hans_df)

        hans = hans.filter(lambda example: example['label'] in [0, 1])

        hans = hans.map(lambda x: tokenizer(x['s1'], x['s2'],
                                                    truncation=False,
                                                    padding=True), 
                                        batched=True, batch_size=batch_size)

        hans.set_format(type='torch', columns=['input_ids', 
                                                'token_type_ids',
                                                'attention_mask', 
                                                'label'])

        dataloader_hans = torch.utils.data.DataLoader(
                hans, 
                batch_size=batch_size)

        self.output_data_dict.update({description: dataloader_hans})


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
            do_lower_case=True)

    eval_data_list = [{'description': 'esnli', 
                        'split_name': 'test', 
                        'premise_name': 'premise', 
                        'hypothesis_name': 'hypothesis', 
                        'explanation_name': 'explanation_1'},
                        {'description': 'esnli', 
                        'explanation_name': 'explanation_1', 
                        'split_name': 'validation', 
                        'premise_name': 'premise', 
                        'hypothesis_name': 'hypothesis'}]

    params_training = get_args()

    dataset_obj = TransformersData(tokenizer, eval_data_list,  params_training)

    print(dataset_obj.eval_returned_dataloaders_dict)
    
