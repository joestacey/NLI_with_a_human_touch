

import os
import csv
import pickle
import json
import numpy as np
import pandas as pd
import transformers
import torch
import random
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, EncoderDecoderModel
from datasets import load_dataset, Dataset
import sys
import argparse
from datetime import datetime

from data import GetValData, TransformersData
from run_files import RunAttention
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
            "--lambda_val", type=float, default=1.0, 
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

    # Arguments concerning which attention heads we are supervising:
    parser.add_argument(
            "--attention_layer", type=int, default=11, 
            help="Self attention layer that we are supervising")
    parser.add_argument(
            "--attention_heads", type=int, default=12, 
            help="The number of heads to apply attention supervision to")
    parser.add_argument(
            "--attention_head_numbers", type=str, default='default', 
            help="Which attention heads to supervise. Provide as 1,2,3,4 etc.\
        The number specified must match the attention_heads parameter. \
        By default the first n attention heads will be supervised \
        (n from attention_heads parameter)")

    params, _ = parser.parse_known_args()
  
    return params


params = get_args()

# We set the attention_head_numbers which we want to supervise:
if params.attention_head_numbers == 'default':
    params.attention_head_numbers = list(
            range(params.attention_heads))
else:
    params.attention_head_numbers = params.attention_head_numbers.split(",")
    params.attention_head_numbers = \
            [int(x) for x in params.attention_head_numbers]

assert len(params.attention_head_numbers) == params.attention_heads, \
        "Incorrect number of attention heads specified"

# We create booleans for some params arguments
params.load_model = bool(params.load_model)
params.randomise = bool(params.randomise)

# Logging file
log_file_name = 'log_attention_models_' + params.train_mode + "_" \
        + str(os.getpid()) + '.txt'
model_log = Log(log_file_name, params)

# If evaluating only, do this for one epoch
if params.train_mode == 'eval':
    params.epochs = 1

# We set the desription used when we save our model
params.description = params.description  \
        + "___pid_" + str(os.getpid()) \
        + "___batch_size_" + str(params.batch_size) \
        + "___att_layer_used_" + str(params.attention_layer) \
        + "___lambda_" + str(params.lambda_val) \
        + "___att_heads_trained_" + str(params.attention_heads) \
        + "___train_mode_" + str(params.train_mode) \
        + "___random_seed_" + str(params.random_seed) \
        + "___model_type_" + str(params.model_type)

params.description = params.description.replace("/", "")

# Set CUDAS to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the type of model we're using
if params.model_type[:4] == 'bert':
    model_category = 'bert'
elif params.model_type[:9] == 'microsoft':
    model_category = 'deberta'


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
        + params.explanation_type + "_all_sentences_saved_" \
        + params.attention_type + ".txt", "rb") as fp:
    weights_all_sentences = pickle.load(fp)

model = AutoModelForSequenceClassification.from_pretrained(
        params.model_type,
        num_labels = 3,
        output_attentions = True,
        output_hidden_states = True)

if params.load_model:
    print("\n \n Loading model \n \n")
    model.load_state_dict(torch.load(params.load_file))

# Create optimiser
optimizer = AdamW(model.parameters(), lr=params.learning_rate)

model.to(device)

run = RunAttention(model, dataset_obj, optimizer, params, model_log, 
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

    # We evaluate for the eSNLI dev data
    for dataset_name, dataset in dataset_obj.dev_dataloader_dict.items():
        print("Dataset:", dataset_name)
        run.evaluate_model(dataset, 'eSNLI version of SNLI dev set')

    # We evaluate for the eSNLI test data
    for dataset_name, dataset in dataset_obj.test_dataloader_dict.items():
        print("Dataset:", dataset_name)
        run.evaluate_model(dataset, 'eSNLI version of SNLI test set')
    
    # We evaluate on other evaluation datasets
    for dataset_name, dataset in dataset_obj.eval_returned_dataloaders_dict.items():
        run.evaluate_model(dataset, dataset_name)

