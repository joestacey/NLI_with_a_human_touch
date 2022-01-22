

import os
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
import re
from data import TransformersData
import spacy
from datasets import load_dataset, Dataset

import argparse

def get_args():

    parser = argparse.ArgumentParser(description="Training model parameters")

    parser.add_argument(
            "--dataset", type=str, default="esnli", 
            help="esnli only supported")
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
            "--attention_type", type=str, default="original", 
            help="'original', or 'reduced' if only considering explanations\
                    with important words in both sentences")
    parser.add_argument(
            "--explanation_type", type=str, default="fullexplanations", 
            help="To use 'fullexplanations' or 'annotated' explanations")
    parser.add_argument(
            "--data_type", type=str, default="train", 
            help="train or test")
    params, _ = parser.parse_known_args()

    return params


class CreateAttentionWeights():

    """
    Creating dictionaries of desired attention weights for each observation
 
    Attributes:
        train_data_name: name of dataset
        model_type: type of model, bert or deberta
        explanation_type: fullexplanations or annotated explanations
        tokenizer_gpt2: gpt2 tokenizer (required for deberta)
        data_type: 'train' for training data
        weight_type: 'original', or 'reduced' if requiring annotated
            ... explanations to highlight words in both sentences
        special_chars: special chars to be separated from neighbouring words
        tokenizer: tokenizer
        dataset: dataset where we want to find the words to attend to
        weights: tokens to attend to for each observation, with input_id lookup
        stop_words: list of stopwords to not pay attention to when matching
            .. to free text explanations

    """

    def __init__(
            self, 
            tokenizer, 
            weight_type: str, 
            dataset: Dataset, 
            model_type: str, 
            train_data_name: str, 
            explanation_type: str, 
            data_type: str):

        self.train_data_name = train_data_name
        if model_type[:4] == "bert":
            self.model_type = "bert"
        elif model_type[:9] == "microsoft":
            self.model_type = "deberta"

        self.explanation_type = explanation_type
        self.tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2')
        self.data_type = data_type
        self.weight_type = weight_type
        self.special_chars = [".", ",", "!", "?", ":", ";", ")", "(", ">", 
                "<", "[", "]", '"']
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        # Weights if using training data
        self.weights = {}

        self.stop_words = ['their', 'i', 'a', 'is', 'are', 'it', 'be', 'by', 
                'for', 'and', 'its', 'an', 'the', 'if', 'of', 'to', 'in', 
                'on', 'all', 'at', 'so', 'too', 'his', 'her', 'there', 
                'just', 'they', 'do', 'what', 'with', 'get', 'as',
              'or', 'has', 'than'] + self.special_chars

        if self.data_type == 'train':
            # Creating attention weights
            self.weights = self.find_desired_words_to_attend_to(
                    'explanation_1')
        
            # Saving our attention weights
            print("Saving attention weights")
            self.save_train_attention_weights()


    def save_train_attention_weights(self) -> None:
        """
        We save the attention weights as dictionaries

        Returns:
            None
        """

        with open("weights_" + self.model_type + self.train_data_name \
                + self.explanation_type + '_all_sentences_saved_' \
                + self.weight_type + '.txt', "wb") as fp:
            pickle.dump(self.weights, fp)


    def spaces_around_special_chars(
            self, 
            text_string: str) -> str:
        """
        Add a space before and after any special characters

        Args: 
            text_string: text string including special characters

        Returns:
            text_string: text string with spaces around special characters
        """

        for chars in self.special_chars:
            text_string = text_string.replace(chars, " " + chars + " ")

        return text_string


    def remove_spaces_around_special_chars(
            self, 
            word: str):
        """
        Removes spaces around any special chars, reverse of adding these in
        
        """

        for chars in self.special_chars:
            word = word.replace(" " + chars + " ", chars)

        return word


    def lowercase_split_to_list_on_spaces(
            self, 
            text_string: str) -> list:
        """
        We return a list of lower case words from a string sentence

        Args:
            text_string: text to split into a string of words

        Returns:
            text_string_split: list of words
        """
        text_string = text_string.lower()

        text_string_split = self.spaces_around_special_chars(
                text_string).split(" ")
        
        return text_string_split


    def construct_from_annotated(
            self, 
            hypothesis_premise_text: str, 
            explanation: str) -> (list, str):
        """
        Formatting the annotated hypothesis and premise into a sentence 
        ... taking punctuation outside of the *'s for important words
        ... eSNLI includes nearby punctuation within these *'s
        
        We also identify the character locations of the * characters

        Args:
            hypothesis_premise_text: hypothesis and premise text
            explanation: hypothesis and premise with *s around important words

        Returns:
            star_locations: a list of character numbers for * characters
            explanation: updated text explanation
            
        """
        # Formatting on spacing below may be simplified
        if explanation == None:
            explanation = " [SEP] "

        tok_sep = explanation.find("[SEP]")
        first_half = explanation[:tok_sep]
        second_half = explanation[tok_sep+5:]

        # Setting the middle [SEP] token for different types of models
        if self.model_type == 'deberta':
            explanation = first_half + "[SEP]" + second_half

        elif self.model_type == 'bert': 
            explanation = first_half + " [SEP] " + second_half

        # Setting the start and end tokens for differnet types of models
        if self.model_type == "bert":
            explanation = '[CLS] ' + explanation + ' [SEP]'
        
        elif self.model_type == 'deberta':
            explanation = '[CLS]' + explanation + '[SEP]'

        original_encoded = self.tokenizer.encode(
                hypothesis_premise_text, 
                add_special_tokens=False)

        # In eSNLI, punctuation's included within the *s, but we dont want this
        for char in self.special_chars:
            explanation = explanation.replace(" *" + char, " " + char + "*")
            explanation = explanation.replace(char + "* ", "*" + char + " ")
            explanation = explanation.replace(char + "*<", "*" + char + "<")
            explanation = explanation.replace(">*" + char, ">" + char + "*")
            explanation = explanation.replace(char + "*[", "*" + char + "[")
            explanation = explanation.replace("]*" + char, "]" + char + "*")

        # BERT tokens do not include a symbol for spaces
        if self.model_type == 'bert':
            explanation = explanation.replace(" ", "")

        # Find where the star characters are:
        star_locations = [m.start() for m in re.finditer('\*', explanation)]       

        return star_locations, explanation


    def find_word_indices_annotated(
            self, 
            explanation: str, 
            star_locations) -> list:

        """
        We return the character number for words that we want to pay attention
        ... to (excluding * characters when finding character numbers)

        Args:
            explanation: premise and hypothesis text with * characters
            star_locations: character numbers for the locations of * characters

        Returns:
            word_indexes: Character number for start of each word to attend to
        """
        # We now loop through all words mentioned:
        word_indexes = []
        for i in range(int(len(star_locations)/2)):
            
            a = 2*i  #Starting star
            b = 2*i + 1  #Ending star
            start = star_locations[a]
            end = star_locations[b]

            word_length = end - start - 1

            explanation_reduced = explanation[:end+1]

            explanation_reduced = explanation_reduced.replace("*", "")
            
            # We find the word start and the word end character numbers
            word_start = len(explanation_reduced) - word_length
            word_end = len(explanation_reduced) - 1
            word_indexes.append((word_start, word_end))
                
        return word_indexes


    def combine_special_tokens(
            self, 
            token_list: list) -> list:
        """
        Combines [ sep ] and [ cls ] to be [SEP] and [CLS] in token list

        Args: 
            token_list: list of words with punctuation also split out

        Returns:
            token_list: list of words with special tokens reassembled 
        """
        
        ind_list = [idx for idx in range(len(token_list)) \
                if (token_list[idx] == 'cls' or  token_list[idx] == 'sep')]

        ind_list_remove = []
        for i in ind_list:
            ind_list_remove.append(i+1)
            ind_list_remove.append(i-1)

        token_list = [token_list[idx] for idx in range(len(token_list)) \
                if idx not in ind_list_remove]

        token_list = [x if x != 'sep' else '[SEP]' for x in token_list]
        token_list = [x if x != 'cls' else '[CLS]' for x in token_list]

        return token_list


    def find_word_indices(
            self, 
            hypothesis_premise_text: str, 
            words_in_both: list) -> list:

        """
        When using the full text explanation, we find the character number for 
        ... words that we want to pay attention to
        
        Args:
            hypothesis_premise_text: hypothesis and premise with special tokens
            words_in_both: list of words in the explanation and sentences

        Returns:
            word_indexes: list of tuples with start and end character for words
                ... that we want to pay attention to
        """

        sentence_tokens = self.lowercase_split_to_list_on_spaces(
                hypothesis_premise_text)

        sentence_tokens = self.combine_special_tokens(sentence_tokens)
        
        # If we find a match, we consider where the word starts and finishes
        word_indexes = []
        special_token_list = ['[SEP]', '[CLS]']
        for word in words_in_both:
            
            for i, tok in enumerate(sentence_tokens):
                if tok == word and tok not in special_token_list:
                                     
                    sentence_to_word = sentence_tokens[:i+1]
                    sentence_to_word = " ".join(sentence_to_word)
                    sentence_to_word = sentence_to_word + " "
                    sentence_to_word = self.remove_spaces_around_special_chars(
                            sentence_to_word)
                    
                    # Spaces are not included as characters in BERT tokens
                    if self.model_type == 'bert':
                        sentence_to_word = sentence_to_word.replace(" ", "")

                    elif self.model_type == 'deberta':
                        sentence_to_word = sentence_to_word.replace(
                                " [CLS] ", 
                                "[CLS]")
                        sentence_to_word = sentence_to_word.replace(
                                " [SEP]", 
                                "[SEP]")
                        sentence_to_word = sentence_to_word.replace(
                                "[SEP] ", 
                                "[SEP]")

                    if sentence_to_word[-1] == " ":
                        sentence_to_word = sentence_to_word[:-1]
                
                    word_start = len(sentence_to_word) - len(word)
                    word_end = len(sentence_to_word)-1
                    word_indexes.append((word_start, word_end))
        
        return word_indexes


    def get_cls_sep_deberta_tokens(
            self, 
            sentence_text: str) -> (list, str, str):
        """
        Create tokenization with DeBERTa, with CLS and SEP tokens (using GPT2)

        Args:
            sentence_text: text including both hypothesis and premise

        Returns:
            tokens: tokenized hypotheis and premise, with SEP and CLS tokens
            premse: string of the premise
            hypothesis: string of the hypothesis
        """

        # For tokenized hypothesis and premise
        sep_token = sentence_text.find("[SEP]") 
        premise = sentence_text[5:sep_token]
        hypothesis = sentence_text[sep_token + 5:-5]

        token_premise = self.tokenizer_gpt2.tokenize(
                premise, 
                add_special_tokens=False)
        
        token_hypothesis = self.tokenizer_gpt2.tokenize(
                hypothesis, 
                add_special_tokens=False)
        
        tokens = ["[CLS]"] + token_premise + ["[SEP]"] \
                + token_hypothesis + ["[SEP]"]

        return tokens, premise, hypothesis


    def deberta_fix_differences_between_snli_and_esnli(
            self,
            explanation: str,
            premise: str,
            hypothesis: str
            ) -> str:
        """
        eSNLI-annotations (with *s) can change the sentence
        ... in particular the spaces can change. As in DeBERTa the spaces make
        ... a difference to the tokenizaton, we need to resolve these issues.

        Args:
            explanation: hypothesis and premise with *s around important words
            premise: snli premise
            hypothesis: snli hypothesis

        Returns:
            explanation: updated explanation
        """
        # For explanation:                                              
        expl_sep_token = explanation.find("[SEP]")                      
        expl_premise = explanation[:expl_sep_token]                     
        expl_hypothesis = explanation[expl_sep_token + 5:]              
        expl_premise = expl_premise.strip(" ")                          
        expl_hypothesis =  expl_hypothesis.strip(" ")                   
                                                                        
        if premise[-1] == " ":                                          
            # We make sure spacing is consistent in both inputs         
            spaces = len(premise) - len(premise.rstrip(" "))            
                                                                        
            for i in range(spaces):                                     
                expl_premise = expl_premise + " "                       
                                                                        
        if hypothesis[-1] == "\\":                                      
            # Backslash characters can be removed at the end of the hypothesis
            expl_hypothesis = expl_hypothesis + "\\"                    
                                                                        
        if hypothesis[0] == " ":                                        
            # We make sure spacing is consistent in both inputs         
            spaces = len(hypothesis) - len(hypothesis.lstrip(" "))      
                                                                        
            for i in range(spaces):                                     
                expl_hypothesis = " " + expl_hypothesis                 
                                                                        
        if premise[0] == " ":                                           
            # We make sure spacing is consistent in both inputs         
            spaces = len(premise) - len(premise.lstrip(" "))            
                                                                        
            for i in range(spaces):                                     
                expl_premise = " " + expl_premise                       
                                                                        
        if hypothesis == "nan":                                         
            # We will not supervise to examples with 'nan' as a hypothesis
            supervise_example = False                                   
                                                                        
        explanation = expl_premise + '[SEP]' + expl_hypothesis

        return explanation


    def token_weights_calc_freetext(
            self,
            words_in_both: list,
            hypothesis_premise_text: str,
            original_encoded: list) -> list:

        """
        Find desired tokens to pay attention to (with 1s and 0s) for example

        Args:
            words_in_both: a list of words in both the explanation and h/p
            hypothesis_premise_text: the hypothesis and premise text
            original_encoded: token ids for the tokenized sentences

        Returns:
            weights: tokens we want attention to be paid to (with 1s and 0s)
        """

        supervise_example = True

        if self.model_type == 'deberta':
            tokens, premise, hypothesis = self.get_cls_sep_deberta_tokens(
                    hypothesis_premise_text)

        else:
            tokens = tokenizer.tokenize(hypothesis_premise_text)

        word_indices = self.find_word_indices(
                hypothesis_premise_text, 
                words_in_both)

        weights = self.get_weights_from_word_indices(
                tokens,
                original_encoded, 
                word_indices, 
                supervise_example)

        return weights


    def get_list_of_token_lens(
            self, 
            tokens: list) -> list:
        """
        Create a list of cumulative token starting characters for word list

        Args:
            tokens: list of words (w/ punctuation and special chars)

        Return:
            token_lens: a list of starting characters for each word in tokens   
        """
        char_len = 0
        token_lens = [0]

        for i, j in enumerate(tokens):
            if self.model_type == 'bert':
                char_len = char_len + len(j.replace("##", ""))
                token_lens.append(char_len)
            else:
                char_len = char_len + len(j)
                # When not bert we may have a space before the token
                if i > 0:
                    token_lens.append(char_len_prev + j.count('Ġ'))
                char_len_prev = char_len

        if self.model_type == 'bert':
            token_lens = token_lens[:-1]

        return token_lens


    def get_weights_from_word_indices(
            self, 
            tokens,
            original_encoded: list, 
            word_indices: list, 
            supervise_example: bool) -> list:
        """
        Creates desired attention weights from word_indices
        
        Args:
            original_encoded: input ids after tokenization
            word_indices: list of tuples showing start and end characters
                ... for words that should be attended to, e.g. [(3, 4), (8, 9)]
            supervise_example: supervise example or not

        Returns:
            weights: A list of 1s and 0s for tokens we want to attend to
        """

        weights = torch.zeros(len(original_encoded))

        token_lens = self.get_list_of_token_lens(tokens)

        # Creating our attention weights
        if supervise_example == True:

            # We create the weights
            for i, j in enumerate(word_indices):
                start, end = j
                for index, cum_length in enumerate(token_lens):
                    if cum_length >= start and cum_length <= end:
                        assert index < len(weights)
                        weights[index] = 1

        # Our weights consist of either 1 or 0 corresponding to each token
        weights = [1 if x >= 1 else 0 for x in weights]

        # Where we don't supervise an example we set the first value to -1
        if supervise_example == False:

            weights = [0]*len(tokens)
            weights[0] = -1

        return weights


    def deberta_further_data_processing(
            hypothesis_premise_text: str, 
            explanation: str, 
            supervise_example: bool) -> bool:
        """
        We do not supervise attention for very small number of special cases
        
        Args:
            hypothesis_premise_text: hypothesis and premise with specal toks
            explanation: hypothesis and premise with *s around important words
            supervise_example: if we want to supervise the example or not

        Returns:
            supervise_example: if we want to supervise a sentence or not
        """

        if explanation.replace("*", "") == '[CLS][SEP][SEP]':
            supervise_example = False

        # We perform additional checks between the two eSNLI inputs
        if supervise_example:

            if hypothesis_premise_text != explanation.replace("*", ""):

                # One sentence already has a star present
                if hypothesis_premise_text.replace("*", "") == \
                        explanation.replace("*", ""):
                    supervise_example = False

        return supervise_example


    def token_weights_calc_annotated(
            self,
            hypothesis_premise_text: str,
            original_encoded: list,
            explanation: str) -> list:
        """
        Find desired tokens to pay attention to (with 1s and 0s) for example

        Args:
            hypothesis_premise_text: hypothesis, premise and special tokens
            orginal_encoded: token ids when original input is tokenized
            explanation: hypothesis and premise text with *s around key words
        
        Returns:
            weights: tokens we want attention to be paid to (with 1s and 0s)
        """

        supervise_example = True

        if self.model_type == 'deberta':
            tokens, premise, hypothesis = self.get_cls_sep_deberta_tokens(
                    hypothesis_premise_text)

            explanation = self.deberta_fix_differences_between_snli_and_esnli(
                        explanation,
                        premise,
                        hypothesis)
        else:
            tokens = tokenizer.tokenize(hypothesis_premise_text)

        
        star_locations, explanation = self.construct_from_annotated(
                hypothesis_premise_text, 
                explanation)

        # Further data processing required for deberta
        if self.model_type == 'deberta':
            supervise_example = self.deberta_further_data_processing(
                hypothesis_premise_text,
                explanation,
                supervise_example)

        word_indices = self.find_word_indices_annotated(
                explanation, 
                star_locations)

        weights = self.get_weights_from_word_indices(
                tokens,
                original_encoded,
                word_indices,
                supervise_example)

        return weights


    def words_in_both_no_stopwords(
            self, 
            list_a: list, 
            list_b:list) -> list:
        """
        Creating a list of words in both list_a and list_b excl. stopwords

        Args:
            list_a: first list of words
            list_b: second list of words

        Returns:
            combined: list of words in both lists (excl. stop words)
        """

        combined = []

        for word in list_a:
            if word in list_b:
                if word not in self.stop_words and word.strip(" ") != "":
                    combined.append(word)

        return combined


    def find_desired_words_to_attend_to(
            self, 
            explanation_name_in_data: str) -> dict:
        """
        Find desired words to attend to for example example
        
        Args:
            explanation_name_in_data: column name in data for the explanation

        Returns:
            desired_att_all_sentences: desired attention pattern for each obs   
        """

        desired_att_all_sentences = {}

        for i in range(len(self.dataset)):
            print("Calculating desired attention weights, example number:",i)

            if self.model_type == 'bert':
                hypothesis_premise_text = "[CLS] " \
                    + self.dataset[i]['premise'] \
                    + " [SEP] " + self.dataset[i]['hypothesis'] \
                    + " [SEP]"
            
            elif self.model_type == "deberta":
                hypothesis_premise_text = "[CLS]" \
                    + self.dataset[i]['premise'] \
                    + "[SEP]" + self.dataset[i]['hypothesis'] \
                    + "[SEP]"
                           
            sentences_word_list = self.lowercase_split_to_list_on_spaces(
                    hypothesis_premise_text)
            explanation = self.dataset[i][explanation_name_in_data]

            # We need to be able to deal with cases where no explanation provided
            if explanation == None:    
                if self.explanation_type == 'fullexplanations':
                    explanation = "  "
                else:
                    explanation = " [SEP] "

            # Input IDs of tokens to be used in the model
            full_tok = self.tokenizer(
                    [(self.dataset[i]['premise'], 
                        self.dataset[i]['hypothesis'])],
                    add_special_tokens=True)
            raw_tokens = full_tok['input_ids'][0]

            # Where we use the text explanation
            if self.explanation_type == 'fullexplanations':

                explanations_word_list = self.lowercase_split_to_list_on_spaces(
                        explanation)
                words_in_both = self.words_in_both_no_stopwords(
                        sentences_word_list, 
                        explanations_word_list)
                # We make this list unique:
                words_in_both = list(dict.fromkeys(words_in_both))
                desired_words_to_attend = self.token_weights_calc_freetext(
                        words_in_both, 
                        hypothesis_premise_text, 
                        raw_tokens)

            elif self.explanation_type == 'annotated':

                desired_words_to_attend = self.token_weights_calc_annotated(
                        hypothesis_premise_text, 
                        raw_tokens, 
                        explanation)

            desired_words_to_attend = self.check_to_remove_attentions(
                        desired_words_to_attend,
                        raw_tokens)

            assert len(raw_tokens) == len(desired_words_to_attend)

            # For explanations we do not want to use, the first value is -1
            if sum(desired_words_to_attend) == 0 \
                    and desired_words_to_attend[0] != -1:
                desired_words_to_attend[0] = -1
                
            desired_att_all_sentences.update(
                    {str(raw_tokens): desired_words_to_attend})
            
        return desired_att_all_sentences


    def check_to_remove_attentions(
            self, 
            weights: list, 
            raw_tokens: list) -> list:
        '''
        We check to see if we should not pay attention to any of our examples 
        
        Args:
            weights: which words we want attention to be paid to (0s and 1s)
            raw_tokens: token ids for premise and hypothesis pair

        Returns:
            weights: updated weights, flagging examples we do not train on
        '''

        if self.model_type == 'bert':
            sep_id = raw_tokens.index(102)
        elif self.model_type == 'deberta':
            sep_id = raw_tokens.index(2)

        premise_weights = weights[:sep_id]
        hypothesis_weights = weights[sep_id:]
        premise_tokens = raw_tokens[:sep_id]
        hypothesis_tokens = raw_tokens[sep_id:]

        # Check if there are too many words with attention in either phrase
        if self.weight_type == 'reduced':
            
            weights = self.check_attention_in_both(
                    premise_weights, 
                    hypothesis_weights, 
                    weights)       

        return weights


    def check_attention_in_both(
            self, 
            premise_weights, 
            hypothesis_weights, 
            weights):
        """
        Checks we want to attend to words in both the hypothesis and premise
        """
        
        if sum(premise_weights) < 1 or sum(hypothesis_weights) < 1:
            weights[0] = -1
            
        return weights


if __name__ == "__main__":

    params_training = get_args()

    print(params_training)

    eval_data_list = []

    # Append SNLI dev and test sets
    eval_data_list.append({'description': 'snli', 'split_name': 'test',
        'premise_name': 'premise', 'hypothesis_name': 'hypothesis'})

    tokenizer = AutoTokenizer.from_pretrained(params_training.model_type)

    weight_type = params_training.attention_type

    dataset_obj = TransformersData(tokenizer,
                eval_data_list,
                params_training)

    print("Explanation type:", params_training.explanation_type)

    if params_training.data_type == 'train':
        
        att_weights = CreateAttentionWeights(
                tokenizer, 
                weight_type,
                dataset_obj.loaded_train_data, 
                params_training.model_type, 
                "_esnli_train_",
                params_training.explanation_type, 
                params_training.data_type)


