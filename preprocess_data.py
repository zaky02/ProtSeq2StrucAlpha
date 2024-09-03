import os
import random
import torch
from SaProt.utils.esm_loader import load_esm_saprot
from SaProt.utils.foldseek_util import get_struc_seq
from transformers import EsmTokenizer, EsmForMaskedLM
import numpy as np
import json

# Load the ESM model and alphabet
#model_path = "models/SaProt_650M_AF2.pt"
#model, alphabet = load_esm_saprot(model_path)
foldseek_path = '/home/phastos/Programs/mambaforge/envs/SaProt/lib/python3.10/site-packages/SaProt/bin/foldseek'
model = EsmForMaskedLM.from_pretrained('westlake-repl/SaProt_650M_AF2')
tokenizer = EsmTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
max_length = 1024

# Function to randomly mask sections of sequences
def mask_sa_vocab(foldseek_seq, mask_token, mask_ratio=0.15, mask_vocab='foldseek'):
    foldseek_seq = list(foldseek_seq)
    num_to_mask = int(int(len(foldseek_seq)/2) * mask_ratio)
    mask_indices = random.sample(range(int(len(foldseek_seq)/2)), num_to_mask)
    if mask_vocab == 'foldseek':
        mask_indices = [(index*2)+1 for index in mask_indices]
    elif mask_vocab == 'sequence':
        mask_indices = [index*2 for index in mask_indices]
    elif mask_vocab == 'any':
        mask_indices = random.sample(range(int(len(foldseek_seq))), num_to_mask)
    else:
        raise ValueError('mask_vocab must be \'foldseek\' or \'sequence\' or \'any\'')
    for index in mask_indices:
        foldseek_seq[index] = mask_token
    return ''.join(foldseek_seq)


# Given a PDB get SA vocab
def get_sa_vocab(pdb_path, chain_id):
    parsed_seqs = get_struc_seq(foldseek_path,
                                pdb_path, [chain_id])[chain_id]
    seq, foldseek_seq, combined_seq = parsed_seqs
    return seq, foldseek_seq, combined_seq
    
    """
    #masked_foldseek_seq = mask_random_sections(foldseek_seq, alphabet.get_tok(alphabet.mask_idx))

    # Prepare sequences for model
    esm_tokens = alphabet.encode(seq)
    esm_tokens = {k: torch.tensor(v).to(device) for k, v in esm_tokens.items()}

    return esm_tokens, masked_foldseek_seq, foldseek_seq
    """


# Function to process all PDB files in a given directory
def process_directory(directory):
    esm_sequences = []
    masked_structural_sequences = []
    target_structural_sequences = []


    import glob
    pdbs = glob.glob('%s*.pdb' %directory)
    chain_id = 'A'
    combined_seqs = []
    for pdb in pdbs:
        seq, foldseek_seq, combined_seq = get_sa_vocab(pdb, chain_id)
        combined_seqs.append(combined_seq)
    print(combined_seqs)
    #tokens = tokenizer.tokenize(combined_seqs)
    inputs = tokenizer(combined_seqs, return_tensors='pt', padding=True, truncation=True, max_length=10)
    print(inputs)

    exit()
    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".pdb"):
            pdb_path = os.path.join(directory, filename)
            chain_id = "A"  # Change if other chains should be processed
            #esm_tokens, masked_foldseek_seq, foldseek_seq = preprocess_protein(pdb_path, chain_id)
            seq, foldseek_seq, combined_seq = get_sa_vocab(pdb_path, chain_id)
            combined_seq_mask = mask_sa_vocab(combined_seq, mask_token='#')
            #print(combined_seq_mask)
            
            tokens = tokenizer.tokenize(combined_seq)
            inputs = tokenizer(combined_seq, return_tensors="pt")
            print(tokens)
            print(inputs)
            tokens = tokenizer.tokenize(combined_seq_mask)
            inputs = tokenizer(combined_seq_mask, return_tensors="pt")
            print(tokens)
            print(inputs)
            
            #mask_candi = [i for i, t in enumerate(tokens) if t[-1] != "#"]
            #mask_num = int(len(mask_candi) * 0.15)
            #print(mask_candi)
            #print(mask_num)
            #mask_idx = np.random.choice(mask_candi, mask_num, replace=False)
            #print(mask_idx)
            #for i in mask_idx:
            #    tokens[i] = tokens[i][:-1] + "#"
            #print(tokens)
            #seq = "".join(tokens)
            #print(seq)
            #tokens = tokenizer.tokenize(seq)[:max_length]
            #print(tokens)
            #seq = " ".join(tokens)
            #print(seq)
            #coords = data['coords'][:max_length] if True else None
            #print(coords)
            #exit()
            exit()
            """
            masked_structural_tokens = alphabet.encode(masked_foldseek_seq)
            masked_structural_tokens = {k: torch.tensor(v).to(device) for k, v in masked_structural_tokens.items()}

            target_structural_tokens = alphabet.encode(foldseek_seq)
            target_structural_tokens = {k: torch.tensor(v).to(device) for k, v in target_structural_tokens.items()}

            esm_sequences.append(esm_tokens)
            masked_structural_sequences.append(masked_structural_tokens)
            target_structural_sequences.append(targetenc_structural_tokens)
            """ 
    return esm_sequences, masked_structural_sequences, target_structural_sequences


# Preprocess data from the structures directory
structures_directory = "structures/"
esm_sequences, masked_structural_sequences, target_structural_sequences = process_directory(structures_directory)

"""
# Save preprocessed data
torch.save({
    'esm_sequences': esm_sequences,
    'masked_structural_sequences': masked_structural_sequences,
    'target_structural_sequences': target_structural_sequences
}, 'preprocessed_data.pth')

data = torch.load('preprocessed_data.pth')
esm_sequences = data['esm_sequences']
print(esm_sequences)
masked_structural_sequences = data['masked_structural_sequences']
print(masked_structural_sequences)
target_structural_sequences = data['target_structural_sequences']
print(target_structural_sequences)
"""
