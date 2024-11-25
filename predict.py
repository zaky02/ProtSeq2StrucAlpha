import torch
import json
import glob
import pandas as pd
from model import TransformerModel
from tokenizer import SequenceTokenizer, FoldSeekTokenizer
from utils.foldseek import get_struc_seq
import numpy as np
from Bio import SeqIO

torch.manual_seed(1234)

def infer(model,
          tokenizer_aa_seqs,
          tokenizer_struc_seqs,
          sequences,
          device='cuda'):
    """
    Perform inference on a batch of amino acid sequences which will
    give us a list of predicted foldseek structural sequences.
    """
    model.eval()

    cls_id = tokenizer_struc_seqs.cls_id
    pad_id = tokenizer_struc_seqs.pad_id

    predicted_struc_seqs = []

    with torch.no_grad():
        for sequence in sequences:
            max_len = len(sequence)

            # Tokenize the input amino acid sequence
            input_ids = tokenizer_aa_seqs(sequence,
                                          max_len=max_len)
            encoder_input_ids = (input_ids['input_ids']).to(device)
            encoder_attention_mask = (input_ids['attention_mask']).to(device)
            
            memory = model.encoder_block(encoder_input=encoder_input_ids,
                                         encoder_padding_mask=encoder_attention_mask)
            
            # Initialise decoder input with the <cls> token
            decoder_input = (torch.tensor([[cls_id]])).to(device)
            predicted_tokens = []

            # Autoregressive decoding
            for _ in range(max_len):
                decoder_padding_mask = (decoder_input == pad_id).to(device)
                logits = model.decoder_block(decoder_input=decoder_input,
                                             memory=memory,
                                             decoder_padding_mask=decoder_padding_mask,
                                             memory_key_padding_mask=encoder_attention_mask)

                # Get the predicted token and append it
                pred_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                predicted_tokens.append(pred_token)
                decoder_input = torch.cat((decoder_input, pred_token), dim=1)

            # Convert predicted tokens to sequence
            predicted_tokens = torch.cat(predicted_tokens, dim=1).squeeze(0).cpu().numpy()
            predicted_struc_seq = [tokenizer_struc_seqs.id2token[id] for id in predicted_tokens]
            predicted_struc_seq = ''.join(predicted_struc_seq)
            predicted_struc_seqs.append(predicted_struc_seq)

    return predicted_struc_seqs


def main(seqs, confile, device=None):
    
    # Load configuration
    with open(confile, 'r') as f:
        config = json.load(f)
    
    # Load and extract sequences from fasta file
    sequences = []
    titles = []
    for record in SeqIO.parse(seqs, "fasta"):
        sequences.append(str(record.seq))
        titles.append(record.id)

    # Initialize tokenizers
    tokenizer_aa_seqs = SequenceTokenizer()
    tokenizer_struc_seqs = FoldSeekTokenizer()
    
    # Initialize device and model
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = TransformerModel(input_dim=tokenizer_aa_seqs.vocab_size,
                             output_dim=tokenizer_struc_seqs.vocab_size,
                             max_len=config['max_len'],
                             dim_model=config['dim_model'],
                             num_heads=config['num_heads'],
                             num_layers=config['num_layers'],
                             ff_hidden_layer=config['ff_hidden_layer'],
                             dropout=config['dropout']).to(device)
 
    weights_path = config['weight_path']
    state_dict = torch.load(weights_path)
    state_dict = {key: value for key, value in state_dict['model'].items()}
    model.load_state_dict(state_dict)
    
    # Perform batch inference
    predicted_struc_seqs = infer(model,
                                 tokenizer_aa_seqs,
                                 tokenizer_struc_seqs,
                                 sequences,
                                 device)

    # Output results
    for aa_seq, pred_struc_seq in zip(sequences,
                                      predicted_struc_seqs):
        print(f"Amino Acid Sequence: {aa_seq}\n")
        print(f"Predicted Structural Sequence: {pred_struc_seq}\n")
        print('---------------------------------------------\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch inference script\
                                                  for Transformer model')
    parser.add_argument('--config',
                        help='Path to configuration file',
                        required=True)
    parser.add_argument('--seqs',
                          help='Sequences to predict foldseek vocab\
                          in fasta format',
                          required=True)
    parser.add_argument('--device',
                        help='either cpu or cuda for gpu',
                        default=None)
    args = parser.parse_args()

    main(seqs=args.seqs, confile=args.config, device=args.device)

