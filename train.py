import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torchinfo import summary
import torchvision
from torchview import draw_graph
from lightning.fabric import Fabric
import random
import json
import glob
import wandb
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
from model import TransformerModel
from utils.timer import Timer
from utils.foldseek import get_struc_seq
from tokenizer import SequenceTokenizer, FoldSeekTokenizer
from dataset import SeqsDataset, collate_fn
from utils import memory as mem

torch.manual_seed(1234)

np.set_printoptions(threshold=999999999)

def train_model(model,
                train_loader,
                optimizer,
                criterion,
                tokenizer_struc_seqs,
                fabric,
                masking_ratio,
                epsilon,
                device='cuda',
                verbose=0):
    """
    Train the model using the specified hyperparamaters

    Args:
        model (model class ...): ....
        train_loader (DataLoader): ...
        optimizer (...): ...
        criterion (...): ...
        epochs (int): Number of epochs
        device (...): ...
        verbose (int): ...
    """
    model.train()
    total_loss = 0.0
    for i, batch in enumerate(train_loader):

        if verbose > 0 and fabric.is_global_zero:
            print(f"\tT.Batch {i+1} of {len(train_loader)} with size {batch['encoder_input_ids'].shape[0]}")

        encoder_input_ids = fabric.to_device(batch['encoder_input_ids'])
        encoder_attention_mask = fabric.to_device(batch['encoder_attention_mask'])
        decoder_input_ids = fabric.to_device(batch['decoder_input_ids'])
        decoder_attention_mask = fabric.to_device(batch['decoder_attention_mask'])
        labels = fabric.to_device(batch['labels'])
    
        optimizer.zero_grad()

        # Forward pass through the model
        logits = model(encoder_input=encoder_input_ids,
                       decoder_input=decoder_input_ids,
                       encoder_padding_mask=encoder_attention_mask,
                       decoder_padding_mask=decoder_attention_mask)
        
        if logits.isnan().any().item():
            raise ValueError('NaN values in logits')

        # Flatten logits first two dimensions (concatenate seqs from batch)
        logits = logits.contiguous().view(-1, logits.size(-1))
        # Normalizing the logits
        # Adding an epsilon value to the logits in order to avoid divergence
        # logits = logits / (torch.max(logits, dim=-1, keepdim=True)[0] + epsilon)

        # Flatten masked_labels dimensions (concatenate seqs from batch)
        labels = labels.contiguous().view(-1)

        # Compute batch loss
        loss = criterion(logits, labels)

        # Backward pass and optimization
        fabric.backward(loss)
        optimizer.step()

        total_loss += loss.item()

        if verbose > 0 and fabric.is_global_zero:
            print(f"\tTraining Average Batch Loss: {loss.item():.4f}")
            print('\t-----------------------')

    avg_loss = total_loss / len(train_loader)
    print(f"Training Average Loss between Batches: {avg_loss:.4f}")
    print(f"Total Training Loss between Batches: {total_loss:.4f}")

    return {'train_loss': avg_loss}

def evaluate_model(model,
                   test_loader,
                   criterion,
                   tokenizer_struc_seqs,
                   fabric,
                   masking_ratio,
                   epsilon,
                   device='cuda',
                   verbose=0):
    """
    Evaluate the model on the test dataset with masking and proper\
    logits processing.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            if verbose > 0 and fabric.is_global_zero:
                print(f"\tE.Batch {i+1} of {len(test_loader)} with size {batch['encoder_input_ids'].shape[0]}")

            encoder_input_ids = fabric.to_device(batch['encoder_input_ids'])
            encoder_attention_mask = fabric.to_device(batch['encoder_attention_mask'])
            labels = fabric.to_device(batch['labels'])

            cls_id = tokenizer_struc_seqs.cls_id
            pad_id = tokenizer_struc_seqs.pad_id
            max_len = encoder_input_ids.shape[-1]
            batch_size = encoder_input_ids.shape[0]

            # Start with the <cls> (start token) as the first input to the decoder
            decoder_input = fabric.to_device(torch.full((batch_size, 1), cls_id))
            predicted = []

            # Forward pass through the encoder
            memory = model.encoder_block(encoder_input=encoder_input_ids,
                                         encoder_padding_mask=encoder_attention_mask)
            
            for t in range(max_len-1):
                # No autoregressive masking; only padding masks are applied
                decoder_padding_mask = fabric.to_device((decoder_input == pad_id))

                # Forward pass through the decoder
                # Memory uses the encoder's padding mask (not sure)
                logits = model.decoder_block(decoder_input=decoder_input,
                                             memory=memory,
                                             decoder_padding_mask=decoder_padding_mask,
                                             memory_key_padding_mask=encoder_attention_mask)
                
                # Get the predicted token from the last step
                pred_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                predicted.append(pred_token)
                # Append the predicted token to the decoder input for the next step
                decoder_input = torch.cat((decoder_input, pred_token), dim=1)
            
            # Concatenate the list of predictions
            predicted = torch.cat(predicted, dim=1)  # shape: (batch_size, trg_len)
            logits = logits.view(-1, logits.shape[-1])  # Flatten the output for loss calculation
            labels = labels.contiguous().view(-1)  # Flatten the target
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predicted = predicted.contiguous().view(-1)

            # Append predictions and labels for F1 calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            correct = (predicted == labels).sum().item()

            if verbose > 0 and fabric.is_global_zero:
                print(f"\tEvaluation Average Batch Loss: {loss.item():.4f}")
                print(f"\tEvalutation correctly predicted structural tokens {correct}/{len(labels)}")
                print('\t-----------------------')

    # Compute F1 score, precision, and recall using sklearn
    precision = precision_score(all_labels,
                                all_preds,
                                average='macro',
                                zero_division=0)
    recall = recall_score(all_labels,
                          all_preds,
                          average='macro',
                          zero_division=0)
    f1 = f1_score(all_labels,
                  all_preds,
                  average='macro',
                  zero_division=0)
    accuracy = accuracy_score(all_labels,
                              all_preds)
    
    avg_loss = total_loss / len(test_loader)
    print(f"Evaluation Average Loss between Batches: {avg_loss:.4f}")
    print(f"Total Evaluation Loss between Batches: {total_loss:.4f}")
    print(f"Evaluation precision {precision:.4f}")
    print(f"Evaluation recall {recall:.4f}")
    print(f"Evaluation accuracy {accuracy:.4f}")
    print(f"Evaluation F1-score {f1:.4f}")

    return {"eval_loss": avg_loss, "precision": precision,
            "recall": recall, "accuracy": accuracy, "f1_score": f1}

def draw_model_graph(model,
                     encoder_tokenizer,
                     decoder_tokenizer,
                     batch_size,
                     max_len,
                     fabric):
    encoder_input = torch.randint(0, encoder_tokenizer.vocab_size,
                                  (batch_size, max_len),
                                  dtype=torch.long)
    decoder_input = torch.randint(0, decoder_tokenizer.vocab_size,
                                  (batch_size, max_len),
                                  dtype=torch.long)

    encoder_input = fabric.to_device(encoder_input)
    decoder_input = fabric.to_device(decoder_input)

    model_graph = draw_graph(model,
                             input_data=[encoder_input, decoder_input],
                             expand_nested=True)

    model_graph.visual_graph.render("model_graph", format="pdf")

    encoder_input = encoder_input.detach().cpu()
    decoder_input = decoder_input.detach().cpu()
    torch.cuda.empty_cache()


def main(confile): 

    with open(confile, 'r') as f:
        config = json.load(f)
 
    verbose = config['verbose']
    if not isinstance(verbose, int):
        raise ValueError('verbose must be set to 0, 1, or 2')
    elif verbose < 0 or verbose > 2:
        raise ValueError('verboe must be set to 0, 1, or 2')

    # Initialize Fabric parallelization
    num_gpus = config['num_gpus']
    parallel_strategy = config['parallel_strategy']
    fabric = Fabric(accelerator='cuda',
                    devices=num_gpus,
                    num_nodes=1,
                    strategy=parallel_strategy)
    
    # Get the data
    structures_dir = config["data_path"]
    pdbs = glob.glob('%s*.pdb' % structures_dir)
    pdbs = pdbs[:1000]

    # Get protein sequence and structural sequence (FoldSeeq) from raw data
    foldseek_path = config["foldseek_path"]
    raw_data = [get_struc_seq(foldseek_path, pdb, chains=['A'])['A'] for pdb in pdbs]
    aa_seqs = [pdb[0] for pdb in raw_data]
    struc_seqs = [pdb[1] for pdb in raw_data]
    if verbose > 0 and fabric.is_global_zero:
        print('- Total amount of structres given %d' %len(aa_seqs))

    # Load Dataset
    tokenizer_aa_seqs = SequenceTokenizer()
    tokenizer_struc_seqs = FoldSeekTokenizer()
    dataset = SeqsDataset(aa_seqs, struc_seqs)

    # Split Dataset into training and testing
    test_split = config["test_split"]
    masking_ratio = config['masking_ratio']
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    if verbose > 0 and fabric.is_global_zero:
        print('- Total amount of tructures in training dataset %d' % len(train_dataset))
        print('- Total amount of structres in testing dataset %d' % len(test_dataset))

    fabric.launch()
    
    # Load DataLoader
    batch_size = config['batch_size']
    max_len = config['max_len']
    train_loader =  DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               collate_fn=lambda batch: collate_fn(batch,
                                                                   tokenizer_aa_seqs,
                                                                   tokenizer_struc_seqs,
                                                                   masking_ratio=masking_ratio,
                                                                   max_len=max_len))

    test_loader =  DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=lambda batch: collate_fn(batch,
                                                                  tokenizer_aa_seqs,
                                                                  tokenizer_struc_seqs,
                                                                  max_len=max_len))
    
    train_loader, test_loader = fabric.setup_dataloaders(train_loader,
                                                         test_loader)

    # Get model hyperparamaters
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    epsilon = config["epsilon"]
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
                             verbose=verbose)#.to('cuda')
     
    model = fabric.setup_module(model)

    draw_model = config['draw_model']
    if draw_model:
        draw_model_graph(model=model,
                         encoder_tokenizer=tokenizer_aa_seqs, 
                         decoder_tokenizer=tokenizer_struc_seqs,
                         batch_size=batch_size,
                         max_len=max_len,
                         frabric=fabric)
    
    if verbose > 0 and fabric.is_global_zero:
        print('- TransformerModel initialized with\n \
                - max_len %d\n \
                - dim_model %d\n \
                - num_heads %d\n \
                - num_layers %d\n \
                - ff_hidden_layer %d\n \
                - dropout %f\n' % (max_len, dim_model, num_heads,
                                   num_layers, ff_hidden_layer, dropout))
    if verbose > 0 and fabric.is_global_zero:
        summary(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
   
    optimizer = fabric.setup_optimizers(optimizer)

    # Print model's state_dict
    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # Print optimizer's state_dict
    #print("Optimizer's state_dict:")
    #for var_name in optimizer.state_dict():
    #    print(var_name, "\t", optimizer.state_dict()[var_name])

    #if fabric.is_global_zero:
    #    print('Model optimizer and DataLoaders to fabric:')
    #    mem.get_GPU_memory(device='cuda:0')
    #    mem.get_GPU_memory(device='cuda:1')
    #    mem.get_CPU_memory()
    #    print('-----------------------------------------')
    
    # Initialize wandb
    if config['get_wandb'] and fabric.is_global_zero:
        wandb.init(project=config["wandb_project"],
                   config={"dataset": "sample_DB",
                           "architecture": "Transformer",
                           "learning_rate": learning_rate,
                           "batch_size": batch_size,
                           "num_epochs": epochs,
                           "dim_model": dim_model,
                           "num_heads": num_heads,
                           "ff_hidden_layer": ff_hidden_layer,
                           "dropout": dropout,
                           "num_layers": num_layers})

    timer = Timer(autoreset=True)
    timer.start('Training/Evaluation (%d epochs)' % epochs)
    for epoch in range(epochs):
        
        timer_epoch = Timer(autoreset=True)
        timer_epoch.start('Epoch %d / %d' %(epoch+1, epochs))

        timer_train = Timer(autoreset=True)
        timer_train.start('Training')
        # Train the model
        training_metrics = train_model(model,
                                       train_loader,
                                       optimizer,
                                       criterion,
                                       tokenizer_struc_seqs,
                                       fabric,
                                       masking_ratio=masking_ratio,
                                       epsilon=epsilon,
                                       device='cuda',
                                       verbose=verbose)
        timer_train.stop()
        
        timer_eval = Timer(autoreset=True)
        timer_eval.start('Evaluation')
        # Evaluate the model
        evaluation_metrics = evaluate_model(model,
                                            test_loader,
                                            criterion,
                                            tokenizer_struc_seqs,
                                            fabric,
                                            masking_ratio=masking_ratio,
                                            epsilon=epsilon,
                                            device='cuda',
                                            verbose=verbose)
        timer_eval.stop()
        
        # Log training and evaluation metrics to wandb
        if config['get_wandb'] and fabric.is_global_zero:
            wandb.log({"train_loss": training_metrics['train_loss'],
                       "eval_loss": evaluation_metrics['eval_loss'],
                       "precision": evaluation_metrics['precision'],
                       "accuracy": evaluation_metrics['accuracy'],
                       "F1": evaluation_metrics['f1_score']},
                       step=epoch+1)

        timer_epoch.stop()
        
    timer.stop('Training/Evaluation (%d epochs) ended' % epochs)


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
