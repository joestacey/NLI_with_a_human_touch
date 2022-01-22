
import os
from data import GetValData, TransformersData, AddSnliHard, AddHANS

def add_ood_datasets(dataset_obj, tokenizer, params):

    dataset_obj = _add_snli_hard(dataset_obj, tokenizer, params)
    dataset_obj = _add_hans(dataset_obj, tokenizer, params)

    return dataset_obj


def _add_snli_hard(dataset_obj, tokenizer, params):

    # Load SNLI-hard
    snli_hard_dict = AddSnliHard(dataset_obj, 'snli_hard',
            os.getcwd() + "/data/snli_hard.jsonl",
            params,
            tokenizer).output_data_dict

    dataset_obj.eval_returned_dataloaders_dict.update(snli_hard_dict)

    return dataset_obj


def _add_hans(dataset_obj, tokenizer, params):
    # Load HANS
    heuristic_list = ['lexical_overlap', 'subsequence', 'constituent']

    hans_dict = AddHANS(dataset_obj, 'hans',
            os.getcwd() + "/data/heuristics_evaluation_set.txt",
            params,
            tokenizer,
            heuristic_list).output_data_dict

    dataset_obj.eval_returned_dataloaders_dict.update(hans_dict)

    return dataset_obj
