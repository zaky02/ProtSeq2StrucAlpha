import torch
from torch.utils.data import Dataset


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

def collate_fn(batch,
               tokenizer_aa_seqs,
               tokenizer_struc_seqs,
               masking_ratio=None,
               max_len=1024):
    
    aa_seqs = [item[0] for item in batch]
    struc_seqs = [item[1] for item in batch]

    # Tokenize the protein sequences (encoder input)
    encoded_aa_seqs = tokenizer_aa_seqs(aa_seqs, max_len=max_len, padding=True, truncation=True)

    # Tokenize the structural sequences (decoder input/output)
    encoded_struc_seqs = tokenizer_struc_seqs(struc_seqs, max_len=max_len, padding=True, truncation=True)

    if masking_ratio:
        mask_id = tokenizer_struc_seqs.mask_id
        masked_decoder_input_ids = masking_struc_seqs_ids(encoded_struc_seqs['input_ids'],
                                                          tokenizer_struc_seqs,
                                                          masking_ratio=masking_ratio)
        decoder_input_ids = masked_decoder_input_ids
        # Get masked labels
        masked_labels = encoded_struc_seqs['input_ids'].clone()
        mask = masked_decoder_input_ids.clone()
        mask = (mask == mask_id)
        masked_labels[~mask]=-100
        labels = masked_labels
    else:
        decoder_input_ids = encoded_struc_seqs['input_ids']
        labels = decoder_input_ids


    # decoder_input_ids and its mask must be left shifted
    decoder_input_ids = decoder_input_ids[:, :-1]
    decoder_attention_mask = encoded_struc_seqs['attention_mask'][:, :-1]
    # labels must be right shifted
    labels = labels[:, 1:]

    return {
        'encoder_input_ids': encoded_aa_seqs['input_ids'],
        'encoder_attention_mask': encoded_aa_seqs['attention_mask'],
        'decoder_input_ids': decoder_input_ids,
        'decoder_attention_mask': decoder_attention_mask,
        'labels':  labels
    }

def masking_struc_seqs_ids(decoder_input_ids,
                           tokenizer_struc_seqs,
                           masking_ratio=0.15):

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
