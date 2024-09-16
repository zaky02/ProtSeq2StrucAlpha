import torch
import itertools
from Bio import SeqIO
from utils.foldseek import get_struc_seq
from SaProt.utils.foldseek_util import get_struc_seq

seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
# max_length refers to aa sequence length no to input length
# with cls and eos the input max size/length is 1026 (+2)
max_len = 1024

foldseek_path = '/home/phastos/Programs/mambaforge/envs/ProtSeq2StrucAlpha/lib/python3.10/site-packages/SaProt/bin/foldseek'

class SaProtTokenizer:
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        self.mask_token = '<mask>'
        self.build_vocab()

    def build_vocab(self):
        self.tokens = [self.cls_token,
                       self.pad_token,
                       self.eos_token,
                       self.unk_token,
                       self.mask_token]

        for seq_token, struc_token in itertools.product(seq_vocab,
                                                        foldseek_struc_vocab):
            token = seq_token + struc_token
            self.tokens.append(token)

        self.vocab_size = len(self.tokens)
        self.token2id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.unk_idx = self.token2id[self.unk_token]
        self.pad_idx = self.token2id[self.pad_token]
        self.cls_idx = self.token2id[self.cls_token]
        self.mask_idx = self.token2id[self.mask_token]
        self.eos_idx = self.token2id[self.eos_token]


    def __call__(self, pdb_list,
                 truncation=True,
                 padding=True,
                 max_len=max_len,
                 return_tensors='pt'):
    
        input_ids = []
        attention_masks = []

        sas = [self.get_sa_vocab(pdb)[2] for pdb in pdbs]
        # add 2 to the lonegst to account for cls and eos
        longest = int(max(len(c) for c in sas)/2) + 2

        for sa in sas:
            sa_list = [sa[i:i+2] for i in range(0, len(sa), 2)]
            print(len(sa_list))

            # Truncation startegy for max_length (not longest)
            if truncation and len(sa_list) > max_len: 
                sa_list = sa_list[:max_len]
                longest = len(sa_list)
            
            sa_list = [self.cls_token] + sa_list + [self.eos_token]
            
            # Padding strategy longest
            if padding and len(sa_list) < longest:
                sa_list = sa_list + [self.pad_token] * (longest - len(sa_list))
            
            input_id = [self.token2id[token] for token in sa_list]
            input_ids.append(input_id)

            attention_mask = [1 if token != self.pad_token else 0 for token in sa_list]
            attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(input_ids)
        attention_masks_tensor = torch.tensor(attention_masks)

        return {'input_ids':input_ids_tensor, 'attention_mask':attention_masks_tensor}

    def get_sa_vocab(self, pdb_path, chain_id='A'):
        parsed_seqs = get_struc_seq(foldseek_path, pdb_path)[chain_id]
        seq, foldseek,  combined = parsed_seqs
        return seq, foldseek, combined

class SequenceTokenizer:
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        self.mask_token = '<mask>'
        self.build_vocab()
    
    def build_vocab(self):
        self.tokens = [self.cls_token,
                       self.pad_token,
                       self.eos_token,
                       self.unk_token,
                       self.mask_token]

        for seq_token in seq_vocab:
            self.tokens.append(seq_token)

        self.vocab_size = len(self.tokens)
        self.token2id = {token: id for id, token in enumerate(self.tokens)}
        self.id2token = {id: token for id, token in enumerate(self.tokens)}

        self.unk_id = self.token2id[self.unk_token]
        self.pad_id = self.token2id[self.pad_token]
        self.cls_id = self.token2id[self.cls_token]
        self.mask_id = self.token2id[self.mask_token]
        self.eos_id = self.token2id[self.eos_token]

    def __call__(self, aa_seqs,
                 truncation=True,
                 padding=True,
                 max_len=max_len,
                 return_tensors='pt'):

        if isinstance(aa_seqs, str):
            aa_seqs = [aa_seqs]
        elif isinstance(aa_seqs, list):
            pass
        else:
            raise ValueError('aa_seqs must be either a single\
                              sequence or a list of sequences')

        input_ids = []
        attention_masks = []
        longest = min(int(max(len(s) for s in aa_seqs)), max_len)
        
        for seq in aa_seqs:
            seq = list(seq)
            
            # Truncation startegy for max_length (not longest)
            if truncation and len(seq) > max_len: 
                seq = seq[:max_len]
            
            seq = [self.cls_token] + seq + [self.eos_token]
            
            # Padding strategy longest
            if padding and len(seq) < longest:
                seq = seq + [self.pad_token] * (longest - len(seq) + 2)
            
            input_id = [self.token2id[token] for token in seq]
            input_ids.append(input_id)
            attention_mask = [1 if token != self.pad_token else 0 for token in seq]
            attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(input_ids)
        attention_masks_tensor = torch.tensor(attention_masks)
        
        return {'input_ids':input_ids_tensor, 'attention_mask':attention_masks_tensor}

    def extract_aa_seq(self, pdb_path, chain_id='A'): 
        with open(pdb_path, 'r') as pdb_file:
            for record in SeqIO.parse(pdb_file, 'pdb-seqres'):
                aa_seq = record.seq
                return aa_seq


class FoldSeekTokenizer:
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        self.mask_token = '<mask>'
        self.build_vocab()
    
    def build_vocab(self):
        self.tokens = [self.cls_token,
                       self.pad_token,
                       self.eos_token,
                       self.unk_token,
                       self.mask_token]

        for struc_token in foldseek_struc_vocab:
            self.tokens.append(struc_token)

        self.vocab_size = len(self.tokens)
        self.token2id = {token: id for id, token in enumerate(self.tokens)}
        self.id2token = {id: token for id, token in enumerate(self.tokens)}

        self.unk_id = self.token2id[self.unk_token]
        self.pad_id = self.token2id[self.pad_token]
        self.cls_id = self.token2id[self.cls_token]
        self.mask_id = self.token2id[self.mask_token]
        self.eos_id = self.token2id[self.eos_token]


    def __call__(self, struc_seqs,
                 truncation=True,
                 padding=True,
                 max_len=max_len,
                 return_tensors='pt'):
 
        if isinstance(struc_seqs, str):
            struc_seqs = [struc_seqs]
        elif isinstance(struc_seqs, list):
            pass
        else:
            raise ValueError('struc_seqs must be either a single\
                              sequence or a list of sequences')
        
        input_ids = []
        attention_masks = []

        longest = min(int(max(len(s) for s in struc_seqs)), max_len)
        
        for seq in struc_seqs:
            seq = list(seq)
            
            # Truncation startegy for max_length (not longest)
            if truncation and len(seq) > max_len: 
                seq = seq[:max_len]
            
            seq = [self.cls_token] + seq + [self.eos_token]
            
            # Padding strategy longest
            if padding and len(seq) < longest:
                seq = seq + [self.pad_token] * (longest - len(seq) + 2)

            input_id = [self.token2id[token] for token in seq]
            input_ids.append(input_id)
            attention_mask = [1 if token != self.pad_token else 0 for token in seq]
            attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(input_ids)
        attention_masks_tensor = torch.tensor(attention_masks)
        
        return {'input_ids':input_ids_tensor, 'attention_mask':attention_masks_tensor}


if __name__ == "__main__":
    import glob

    structures_directory = 'structures/'
    pdbs = glob.glob('%s*.pdb'%structures_directory)
    
    # SaProt tokenizer
    tokenizer_sa = SaProtTokenizer()
    inputs_sa = tokenizer_sa(pdbs)
    print(inputs_sa)

    # ESM tokenizer
    tokenizer_seq = SequenceTokenizer()
    inputs_seq = tokenizer_seq(pdbs)
    print(inputs_seq)

    # FoldSeek tokenizer
    tokenizer_foldseek = FoldSeekTokenizer()
    inputs_foldseek = tokenizer_foldseek(pdbs)
    print(inputs_foldseek)
