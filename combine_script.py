
import pickle
import argparse

def get_args():

  parser = argparse.ArgumentParser(description="Training model parameters")

  # Arguments for modelling different scenarios
  parser.add_argument("--model_type", type=str, default="bert", help="Model to be used")
  params, _ = parser.parse_known_args()

  return params

params_training = get_args()

print(params_training)
# We open up the filenames that we want to combine

str1 = 'weights_' + params_training.model_type + '_esnli_train_fullexplanations_all_sentences_saved_original.txt'
str2 = 'weights_' + params_training.model_type + '_esnli_train_annotated_all_sentences_saved_reduced.txt'
output_str = 'weights_' + params_training.model_type + '_esnli_train_combined_all_sentences_saved_combined.txt'

# We load the weights we want to combine
with open(str1, "rb") as fp:
    weights1 = pickle.load(fp)

with open(str2, "rb") as fp:
    weights2 = pickle.load(fp)

# All the input IDs in either set of weights 
keys1 = list(weights1.keys())
keys2 = list(weights2.keys())

print("Combining training weights")


def combine_weights(weights1, weights2, keys):

    new_weights_dict = {}
    # Creating average weights
    for k in keys:

        w1 = weights1[k]
        w2 = weights2[k]

        # Case 1: both weights to be attended to:
        if weights1[k][0] != -1 and weights2[k][0] != -1:
        
            new_w = [(w1[idx] + w2[idx])/2 for idx in range(len(w1))]
 
            assert len(new_w) == len(w1), "Length error"
            assert len(new_w) == len(w2), "Length error"

        elif weights1[k][0] != -1 and weights2[k][0] == -1:

            new_w = w1

            assert len(new_w) == len(w1), "Length error"
            assert len(new_w) == len(w2), "Length error"

        elif weights1[k][0] == -1 and weights2[k][0] != -1:
    
            new_w = w2

        elif weights1[k][0] == -1 and weights2[k][0] == -1:

            new_w = w1

        new_weights_dict.update({k: new_w})
    
    return new_weights_dict

new_weights_dict = combine_weights(weights1, weights2, keys1)

with open(output_str, "wb") as fp:
            pickle.dump(new_weights_dict, fp)

