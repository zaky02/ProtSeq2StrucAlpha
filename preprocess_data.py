import os
import random
import torch
from SaProt.utils.esm_loader import load_esm_saprot
from SaProt.utils.foldseek_util import get_struc_seq

# Load the ESM model and alphabet
model_path = "/home/cactus/zak/PLM/structures/SaProt_650M_AF2.pt"
model, alphabet = load_esm_saprot(model_path)

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Function to randomly mask sections of sequences
def mask_random_sections(sequence, mask_token, mask_ratio=0.15):
    sequence = list(sequence)
    num_to_mask = int(len(sequence) * mask_ratio)
    mask_indices = random.sample(range(len(sequence)), num_to_mask)
    for idx in mask_indices:
        sequence[idx] = mask_token
    return ''.join(sequence)


# Function to preprocess a single protein structure and generate structural tokens
def preprocess_protein(pdb_path, chain_id):
    parsed_seqs = get_struc_seq("bin/foldseek", pdb_path, [chain_id])[chain_id]
    seq, foldseek_seq, combined_seq = parsed_seqs

    masked_foldseek_seq = mask_random_sections(foldseek_seq, alphabet.get_tok(alphabet.mask_idx))

    # Prepare sequences for model
    esm_tokens = alphabet.encode(seq)
    esm_tokens = {k: torch.tensor(v).to(device) for k, v in esm_tokens.items()}

    return esm_tokens, masked_foldseek_seq, foldseek_seq


# Function to process all PDB files in a given directory
def process_directory(directory):
    esm_sequences = []
    masked_structural_sequences = []
    target_structural_sequences = []

    for filename in os.listdir(directory):
        if filename.endswith(".cif") or filename.endswith(".pdb"):
            pdb_path = os.path.join(directory, filename)
            chain_id = "A"  # Change if other chains should be processed
            try:
                esm_tokens, masked_foldseek_seq, foldseek_seq = preprocess_protein(pdb_path, chain_id)
                masked_structural_tokens = alphabet.encode(masked_foldseek_seq)
                masked_structural_tokens = {k: torch.tensor(v).to(device) for k, v in masked_structural_tokens.items()}

                target_structural_tokens = alphabet.encode(foldseek_seq)
                target_structural_tokens = {k: torch.tensor(v).to(device) for k, v in target_structural_tokens.items()}

                esm_sequences.append(esm_tokens)
                masked_structural_sequences.append(masked_structural_tokens)
                target_structural_sequences.append(target_structural_tokens)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    return esm_sequences, masked_structural_sequences, target_structural_sequences


# Preprocess data from the structures directory
structures_directory = "structures/"
esm_sequences, masked_structural_sequences, target_structural_sequences = process_directory(structures_directory)

# Save preprocessed data
torch.save({
    'esm_sequences': esm_sequences,
    'masked_structural_sequences': masked_structural_sequences,
    'target_structural_sequences': target_structural_sequences
}, 'preprocessed_data.pth')
