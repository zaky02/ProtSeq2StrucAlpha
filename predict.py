import torch
from model import TransformerModel
from tokenizer import SequenceTokenizer, FoldSeekTokenizer
from utils.foldseek import get_struc_seq
import json
import sys

def load_model(config):
    # Initialize model
    model = TransformerModel(input_dim=config['input_dim'],
                             output_dim=config['output_dim'],
                             max_len=config['max_len'],
                             dim_model=config['dim_model'],
                             num_heads=config['num_heads'],
                             num_layers=config['num_layers'],
                             ff_hidden_layer=config['ff_hidden_layer'],
                             dropout=config['dropout'])
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    return model

def infer(model, tokenizer_aa_seqs, tokenizer_struc_seqs, sequence, device='cuda'):
    """
    Perform inference on a single amino acid sequence.
    Args:
        model: Trained Transformer model.
        tokenizer_aa_seqs: Tokenizer for amino acid sequences.
        tokenizer_struc_seqs: Tokenizer for structural sequences.
        sequence (str): Input amino acid sequence.
        device (str): Device to perform inference on.
    Returns:
        predicted_struc_seq (str): Predicted structural sequence.
    """
    cls_id = tokenizer_struc_seqs.cls_id
    pad_id = tokenizer_struc_seqs.pad_id
    max_len = len(sequence)

    # Tokenize amino acid sequence
    encoder_input_ids = torch.tensor(tokenizer_aa_seqs.encode(sequence)).unsqueeze(0).to(device)

    # Start with the <cls> token as the initial decoder input
    decoder_input = torch.full((1, 1), cls_id, dtype=torch.long, device=device)
    predicted_tokens = []

    # Forward pass through encoder
    with torch.no_grad():
        encoder_attention_mask = (encoder_input_ids != pad_id).to(device)
        memory = model.encoder_block(encoder_input=encoder_input_ids, encoder_padding_mask=encoder_attention_mask)

        # Autoregressive decoding
        for _ in range(max_len - 1):
            decoder_padding_mask = (decoder_input == pad_id).to(device)
            logits = model.decoder_block(decoder_input=decoder_input,
                                         memory=memory,
                                         decoder_padding_mask=decoder_padding_mask,
                                         memory_key_padding_mask=encoder_attention_mask)
            # Get predicted token and append it
            pred_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            predicted_tokens.append(pred_token)
            decoder_input = torch.cat((decoder_input, pred_token), dim=1)
    
    # Convert predicted tokens to sequence
    predicted_tokens = torch.cat(predicted_tokens, dim=1).squeeze(0).cpu().numpy()
    predicted_struc_seq = tokenizer_struc_seqs.decode(predicted_tokens)

    return predicted_struc_seq

def main(confile, sequence):
    # Load configuration
    with open(confile, 'r') as f:
        config = json.load(f)

    # Load model and tokenizers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(config).to(device)
    tokenizer_aa_seqs = SequenceTokenizer()
    tokenizer_struc_seqs = FoldSeekTokenizer()

    # Perform inference
    predicted_struc_seq = infer(model, tokenizer_aa_seqs, tokenizer_struc_seqs, sequence, device)
    print(f"Input Sequence: {sequence}")
    print(f"Predicted Structural Sequence: {predicted_struc_seq}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference script for Transformer model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--sequence', type=str, required=True, help='Input amino acid sequence for inference')
    args = parser.parse_args()

    main(confile=args.config, sequence=args.sequence)

