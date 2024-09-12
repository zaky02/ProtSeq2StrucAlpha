import json
import torch
import random
from utils.timer import Timer

# Load hyperparameters from JSON file
"""
input_dim = config["input_dim"]
output_dim = config["output_dim"]
embedding_dim = config["embedding_dim"]
units = config["units"]
batch_size = config["batch_size"]
epochs = config["epochs"]
learning_rate = config["learning_rate"]
"""
"""
# TO DO: apply masking non-randomly by introducing information
# about attention values of the words and mask accordingly. Also, 
# base the masking on pLDDT as found in the SaProt github.

def masking_seq(seq, mask_token, mask_ratio=0.15):
    seq = list(seq)
    seq = [seq[i] + seq[i+1] for i in range(0, len(seq), 2)]
    num_to_mask = int(len(seq) * mask_ratio)
    # randomly select indices from the sequence
    mask_indices = random.sample(range(int(len(seq)/2)), num_to_mask)
    mask_indices = [(i*2)+1 for i in mask_indices]
    for i in mask_indices:
        seq[i] = seq[i][0] + mask_token
    return seq

masking_seq(seq='AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRr', mask_token='#')
"""

def train_model():
    """

    """
    pass

def main(confile=confile):
    """

    """
    
    with open(confile, 'r') as f:
        config = json.load(f)
    
    timer = Timer(autoreset=True)
    timer.start('Training started')
    timer.stop('Training ended')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    args = parser.parse_args()

    confile = args.config

    main(confile=confile)
