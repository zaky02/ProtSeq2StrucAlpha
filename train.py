import json
import torch
import random
import glob
from utils.timer import Timer
from utils.foldseek import get_foldseek_seq

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
    # tokenitzar tot
    # DataStruc DataLoc
    # split train test val
    # ...
    pass

def main(confile):
    """

    """
    
    with open(confile, 'r') as f:
        config = json.load(f)

    structures_dir = config["data_path"]
    foldseek_path = config["foldseek_path"]
    pdbs = glob.glob('%s*.pdb' % structures_dir)
    print(pdbs)
    for pdb in pdbs:
        print(pdb)
        # At the moment only took seq and struc_seq from chain A
        seq, struc_seq = get_foldseek_seq(foldseek_path, pdb, chains=['A'])['A']
        print(seq, struc_seq)
    exit()

    # pillar tots els foldseek seqs i aa seqs
    # carregar resta paramatres config (els del model)
    # carregar wandadb user

    
    timer = Timer(autoreset=True)
    timer.start('Training started')
    #cridar a train_model
    timer.stop('Training ended')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    args = parser.parse_args()

    confile = args.config

    main(confile=confile)
