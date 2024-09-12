import json
import torch
import random
import glob
import wandb
from utils.timer import Timer
from utils.foldseek import get_foldseek_seq
from tokenizer import SequenceTokenizer, FoldSeekTokenizer

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

def train_model(seqs,
                struc_seqs,
                epochs: int = 10,
                learning_rate: float = 0.0001,
                batch_size: int = 10,
                verbose=False):
    """
    Train the model using the specified hyperparamaters

    Args:
        seqs (list): of protein sequences
        struc_seqs (list): corresponding foldseek structural sequences
        epochs: The number of epochs to train the model
        learning_rate: The learning rate
        batch_size: The batch size
    """
    

    # Tokenize protein sequences
    tokenizer_seqs = SequenceTokenizer()

    # Tokenize structural sequences
    tokenizer_foldseek = FoldSeekTokenizer()

    # Split the dataset


def main(confile):
    
    with open(confile, 'r') as f:
        config = json.load(f)

    # Get the data
    structures_dir = config["data_path"]
    pdbs = glob.glob('%s*.pdb' % structures_dir)

    # Get protein sequence and structural sequence (FoldSeeq)
    foldseek_path = config["foldseek_path"]
    data = [get_foldseek_seq(foldseek_path, pdb, chains=['A'])['A'] for pdb in pdbs]
    seqs = [pdb[0] for pdb in data]
    struc_seqs = [pdb[1] for pdb in data]

    # Get hyperparamaters
    get_wandb = config['get_wandb']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']

    # Configure wandb
    if get_wandb:
        wandb.init(
            project=config["wandb_project"],
            config={"dataset": "sample_DB",
                    "architecture": "Transformer"}
        )

    # Train the model
    timer = Timer(autoreset=True)
    timer.start('Training started')
    train_model(seqs,
                struc_seqs,
                epochs,
                learning_rate,
                batch_size,
                verbose=False)
    timer.stop('Training ended')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    args = parser.parse_args()

    confile = args.config

    main(confile=confile)
