import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import random
import json
import glob
import wandb
import numpy as np
import sys
from model import TransformerModel
from utils.timer import Timer
from utils.foldseek import get_struc_seq
from tokenizer import SequenceTokenizer, FoldSeekTokenizer

np.set_printoptions(threshold=sys.maxsize)

class SeqsDataset(Dataset):
    def __init__(self, aa_seqs, struc_seqs):
        self.aa_seqs = aa_seqs
        self.struc_seqs = struc_seqs

    def __len__(self):
        return len(self.aa_seqs)

    def __getitem__(self, idx):
        # Return protein and structural sequence pairs without tokenizing
        aa_seq = self.aa_seqs[idx]
        struc_seq = self.struc_seqs[idx]
        return aa_seq, struc_seq

def collate_fn(batch, tokenizer_aa_seqs, tokenizer_struc_seqs, max_len=1024):
    aa_seqs = [item[0] for item in batch]
    struc_seqs = [item[1] for item in batch]

    # Tokenize the protein sequences (encoder input)
    encoded_aa_seqs = tokenizer_aa_seqs(aa_seqs, max_len=max_len, padding=True, truncation=True)

    # Tokenize the structural sequences (decoder input/output)
    encoded_struc_seqs = tokenizer_struc_seqs(struc_seqs, max_len=max_len, padding=True, truncation=True)

    return {
        'encoder_input_ids': encoded_aa_seqs['input_ids'],
        'encoder_attention_mask': encoded_aa_seqs['attention_mask'],
        'decoder_input_ids': encoded_struc_seqs['input_ids'],
        'decoder_attention_mask': encoded_struc_seqs['attention_mask']
    }


def train_model(model,
                train_loader,
                optimizer,
                criterion,
                tokenizer_struc_seqs,
                masking_ratio,
                device='cuda',
                verbose=False):
    """
    Train the model using the specified hyperparamaters

    Args:
        model (model class ...): ....
        train_loader (DataLoader): ...
        optimizer (...): ...
        criterion (...): ...
        epochs (int): Number of epochs
        device (...): ...
        verbose (bool): ...
    """
    model.train()

    total_loss = 0.0
    for batch in train_loader:
        encoder_input_ids = batch['encoder_input_ids'].to(device)
        encoder_attention_mask = batch['encoder_attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
        
        optimizer.zero_grad()

        masked_decoder_input_ids = masking_struc_seqs_ids(decoder_input_ids,
                                                          tokenizer_struc_seqs,
                                                          masking_ratio)
        # Forward pass through the model
        logits = model(encoder_input=encoder_input_ids,
                        decoder_input=masked_decoder_input_ids,
                        encoder_padding_mask=encoder_attention_mask,
                        decoder_padding_mask=decoder_attention_mask)
        
        print(logits)
        
        # Get masked labels
        masked_labels = decoder_input_ids.clone()
        mask = masked_decoder_input_ids.clone()
        mask_id = tokenizer_struc_seqs.mask_id
        mask = (mask == mask_id)
        masked_labels[~mask] = -100
        # Flatten logits first two dimensions (concatenate seqs from batch)
        logits = logits.view(-1, logits.size(-1)) 
        # Flatten masked_labels dimensions (concatenate seqs from batch)
        masked_labels = masked_labels.view(-1)
        
        # Compute batch loss
        loss = criterion(logits, masked_labels)
        print(loss)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if verbose:
            print(f"Training Average Batch Loss: {loss.item():.4f}")
        exit()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Training Average Loss between Batches: {avg_loss:.4f}")
    print(f"Total Training Loss between Batches: {total_loss:.4f}")
    exit()

def masking_struc_seqs_ids(decoder_input_ids, tokenizer_struc_seqs, masking_ratio=0.15):
    mask_token_id = tokenizer_struc_seqs.mask_id
    eos_token_id = tokenizer_struc_seqs.eos_id

    masked_decoder_input_ids = decoder_input_ids.clone()
    for input_id in masked_decoder_input_ids:
        end_idx = (input_id == eos_token_id).nonzero(as_tuple=True)[0].item()
        seq_len = end_idx - 1
        num_elements_to_mask = max(1, int(seq_len * masking_ratio))
        idxs_to_mask = torch.randperm(seq_len)[:num_elements_to_mask] + 1
        input_id[idxs_to_mask] = mask_token_id
    
    return masked_decoder_input_ids


def evaluate_model(model,
                   test_loader,
                   criterion,
                   device='cuda',
                   verbose=False):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            encoder_input_ids = batch['encoder_input_ids'].to(device)
            encoder_attention_mask = batch['encoder_attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            # Forward pass
            logits = model(encoder_input=encoder_input_ids,
                           decoder_input=decoder_input_ids,
                           encoder_mask=encoder_attention_mask,
                           decoder_mask=decoder_attention_mask,
                           memory_mask=None,
                           encoder_padding_mask=None,
                           decoder_padding_mask=None)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), decoder_input_ids.view(-1))
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(logits, dim=-1)
            total_correct += (predicted == decoder_input_ids).sum().item()
            total_samples += decoder_input_ids.numel()

            if verbose:
                print(f"Processed batch with loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    if verbose:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

    return {"avg_loss": avg_loss, "accuracy": accuracy}

def main(confile):
    
    with open(confile, 'r') as f:
        config = json.load(f)

    verbose = config['verbose']

    # Get the data
    structures_dir = config["data_path"]
    pdbs = glob.glob('%s*.pdb' % structures_dir)
    pdbs = pdbs[:100]

    # Get protein sequence and structural sequence (FoldSeeq) from raw data
    foldseek_path = config["foldseek_path"]
    raw_data = [get_struc_seq(foldseek_path, pdb, chains=['A'])['A'] for pdb in pdbs]
    aa_seqs = [pdb[0] for pdb in raw_data]
    struc_seqs = [pdb[1] for pdb in raw_data]
    if verbose:
        print('- Total amount of structres given %d' %len(aa_seqs))

    # Load Dataset
    tokenizer_aa_seqs = SequenceTokenizer()
    tokenizer_struc_seqs = FoldSeekTokenizer()
    dataset = SeqsDataset(aa_seqs, struc_seqs)

    # Split Dataset into training and testing
    test_split = config["test_split"]
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    if verbose:
        print('- Total amount of tructures in training dataset %d' % len(train_dataset))
        print('- Total amount of structres in testing dataset %d' % len(test_dataset))
    
    # Load DataLoader
    batch_size = config['batch_size']
    max_len = config['max_len']
    train_loader =  DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=lambda batch: collate_fn(batch,
                                                                   tokenizer_aa_seqs,
                                                                   tokenizer_struc_seqs,
                                                                   max_len=max_len)) 
    test_loader =  DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch,
                                                                  tokenizer_aa_seqs,
                                                                  tokenizer_struc_seqs,
                                                                  max_len=max_len))

    # Get model hyperparamaters
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    masking_ratio = config['masking_ratio']
    dim_model = config['dim_model']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    ff_hidden_layer = config['ff_hidden_layer']
    dropout = config['dropout']
    
    # Initialize model, optimizer, and loss function
    model = TransformerModel(input_dim=tokenizer_aa_seqs.vocab_size,
                             output_dim=tokenizer_struc_seqs.vocab_size,
                             max_len=max_len,
                             dim_model=dim_model,
                             num_heads=num_heads,
                             num_layers=num_layers,
                             ff_hidden_layer=ff_hidden_layer,
                             dropout=dropout,
                             verbose=verbose).to('cuda')
    if verbose:
        print('- TransformerModel initialized with\n \
                - max_len %d\n \
                - dim_model %d\n \
                - num_heads %d\n \
                - num_layers %d\n \
                - ff_hidden_layer %d\n \
                - dropout %f\n' % (max_len, dim_model, num_heads,
                                   num_layers, ff_hidden_layer, dropout))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    
    timer = Timer(autoreset=True)
    timer.start('Training started')
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
    
        # Train the model
        train_model(model,
                    train_loader,
                    optimizer,
                    criterion,
                    tokenizer_struc_seqs,
                    masking_ratio=masking_ratio,
                    device='cuda',
                    verbose=verbose)
        exit() 
        # Evaluate the model
        evaluation_results = evaluate_model(model,
                                            test_loader,
                                            criterion,
                                            device='cuda',
                                            verbose=verbose)
        
        print(f"Evaluation Results - Loss: {evaluation_results['avg_loss']}, Accuracy: {evaluation_results['accuracy']}")
        
        # Log the evaluation results to wandb if applicable
        get_wandb = config['get_wandb']
        if get_wandb:
            wandb.init(project=config["wandb_project"],
                       config={"dataset": "sample_DB",
                               "architecture": "Transformer"})
            wandb.log({"epoch": epoch + 1, 
                       "loss": evaluation_results['avg_loss'], 
                       "accuracy": evaluation_results['accuracy']})
    
    timer.stop('Training ended')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str,
                        default='config.json',
                        help='Configuration file',
                        required=True)
    args = parser.parse_args()

    confile = args.config

    main(confile=confile)
