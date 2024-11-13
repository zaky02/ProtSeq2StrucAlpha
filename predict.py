import torch
import json
import glob
from model import TransformerModel
from tokenizer import SequenceTokenizer, FoldSeekTokenizer
from utils.foldseek import get_struc_seq

torch.manual_seed(12345)

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
                                          max_len=max_len,
                                          padding=True,
                                          truncation=True)

            encoder_input_ids = input_ids['input_ids'].to(device)
            encoder_attention_mask = input_ids['attention_mask'].to(device)
            
            memory = model.encoder_block(encoder_input=encoder_input_ids,
                                         encoder_padding_mask=encoder_attention_mask)
            
            # Initialise decoder input with the <cls> token
            decoder_input = torch.tensor([[cls_id]]).to(device)
            predicted_tokens = []

            # Autoregressive decoding
            for _ in range(max_len - 1):
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


def main(confile):
    # Load configuration
    with open(confile, 'r') as f:
        config = json.load(f)

    # Initialize tokenizers
    tokenizer_aa_seqs = SequenceTokenizer()
    tokenizer_struc_seqs = FoldSeekTokenizer()

    # Initialize device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerModel(input_dim=tokenizer_aa_seqs.vocab_size,
                             output_dim=tokenizer_struc_seqs.vocab_size,
                             max_len=config['max_len'],
                             dim_model=config['dim_model'],
                             num_heads=config['num_heads'],
                             num_layers=config['num_layers'],
                             ff_hidden_layer=config['ff_hidden_layer'],
                             dropout=config['dropout']).to(device)
    
    # Load and extract sequences from PDB files
    structures_dir = config['data_path']
    pdbs = glob.glob(f"{structures_dir}/*.pdb")
    pdbs = pdbs[:10]
    sequences = []
    for pdb in pdbs:
        raw_data = get_struc_seq(config["foldseek_path"], pdb, chains=['A'])
        if 'A' in raw_data:
            aa_seq = raw_data['A'][0]
            sequences.append(aa_seq)
        else:
            print(f"Warning: No valid chain 'A' sequence found in {pdb}")

    # Perform batch inference
    predicted_struc_seqs = infer(model,
                                 tokenizer_aa_seqs,
                                 tokenizer_struc_seqs,
                                 sequences,
                                 device)

    # Output results
    for pdb, aa_seq, pred_struc_seq in zip(pdbs,
                                           sequences,
                                           predicted_struc_seqs):
        print(f"Amino Acid Sequence: {aa_seq}\n")
        print(f"Predicted Structural Sequence: {pred_struc_seq}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch inference script\
                                                  for Transformer model')
    parser.add_argument('--config',
                        help='Path to configuration file',
                        required=True)
    args = parser.parse_args()

    main(confile=args.config)

