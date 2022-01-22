# Supervising Model Attention with Human Explanations for Robust Natural Language Inference - Project Code 

This repository contains the code supporting our AAAI-2022 paper Supervising Model Attention with Human Explanations for Robust Natural Language Inference:

![Paper screenshot](https://user-images.githubusercontent.com/51410870/150629722-641ec52a-23c6-4371-affa-b5625e4ffad6.png)


https://arxiv.org/pdf/2104.08142.pdf

Contact details: j.stacey20@imperial.ac.uk

We provide test and dev figures for a specific seed below, as the figures provided in the paper average over 25 random seeds.

We report performance in the paper on the SNLI dev and SNLI test sets from Huggingface, rather than using the SNLI data within e-SNLI (we find minor differences between the two datasets for several examples).

# Instructions

## 1) Downloading data:

The code requires downloading HANS and SNLI_hard and putting these into the data folder. Please run get_ood_data.sh to do this. 

You will also need to download e-SNLI, saving the four CSV files in the dataset_esnli folder (see https://github.com/OanaMariaCamburu/e-SNLI for e-SNLI)

## 2) Creating desired attention weights:

We now create the desired attention weighs, highlighting important words that we would like the model to attend to. 

There are several options, either you can use the free text explanations (setting explanation_type to fullexplanations), or using the highlighted words (setting explanation_type to annotated). 

You can also choose to not consider explanations that would only supervise attention to one of the two sentences (setting attention_type to ‘reduced’).

To create the variations used in the paper, please use:

- **python create_attention_weights.py --explanation_type fullexplanations --attention_type original**

- **python create_attention_weights.py --explanation_type annotated --attention_type original**

- **python create_attention_weights.py --explanation_type annotated --attention_type reduced**

To create the attention weights combining the free text explanations and the highlighted word explanations, please use:

- **python combine_script.py**

## 3) Running the models:

### 3.1) Running the model supervising existing attention weights 

To supervise the three best heads using the existing model attention (see the bottom part of Table 1 in the paper), we use:

- **python attention_model.py --attention_type combined --explanation_type combined --attention_heads 3 --attention_head_numbers 1,2,3 --random_seed 32**

We find the best performance when supervising head numbers 1, 2 and 3.

### 3.2) Running the baseline:

By setting the lambda value to be 0 we run the baseline (top row of Table 1 in the paper).

- **python attention_model.py --attention_type combined --explanation_type combined --lambda_val 0.0 --random_seed 32**

### 3.3) Finding the randomised baseline result:

We set the randomised argument to 1 for our randomised baseline (mentioned in the text in the experiments section):

- **python attention_model.py --attention_type combined --explanation_type combined --random_seed 32 --randomise 1**

### 3.4) Supervising an additional attention layer on top of the model:

To supervise the additional attention layer on top of the model (see middle section of table 1 in the paper):

- **python top_attention_layer.py --attention_type combined --explanation_type combined --random_seed 32 --lambda_val 1.4**

### 3.5) Pruthi et al-adapted method:

To replicate the results from Pruthi et al-adapted (see table 2), we use:

- **python attention_model.py --lambda_val 0.01 --loss kl --attention_type original --explanation_type annotated --random_seed 32**

# Expected results - Seed 32

### 1) Running the model supervising existing attention weights 

- SNLI-test accuracy: 90.22%, SNLI-dev accuracy: 90.50%

### 2) Running the baseline:

SNLI-test accuracy: 89.62%, SNLI-dev accuracy: 89.83%

### 3) Finding the randomised baseline result:

SNLI-test accuracy: 89.31%, SNLI-dev accuracy: 89.1%

### 4) Supervising an additional attention layer on top of the model:

SNLI-test accuracy: 90.1%, SNLI-dev accuracy: 90.22%

### 5) Pruthi et al-adapted method:

SNLI-test accuracy: 90.28%, SNLI-dev accuracy: 90.35%
