import torch
import itertools
from Bio import SeqIO
from SaProt.utils.foldseek_util import get_struc_seq

seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
# max_length refers to aa sequence length no to input length
# with cls and eos the input max size/length is 1026 (+2)
max_length = 1024

foldseek_path = '/home/phastos/Programs/mambaforge/envs/SaProt/lib/python3.10/site-packages/SaProt/bin/foldseek'

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
                 max_length=max_length,
                 return_tensors='pt'):
        
        input_ids = []
        attention_masks = []

        sas = [self.get_sa_vocab(pdb)[2] for pdb in pdbs]
        # add 2 to the lonegst to account for cls and eos
        longest = int(max(len(c) for c in sas)/2) + 2

        for sa in sas:
            sa_list = [sa[i:i+2] for i in range(0, len(sa), 2)]

            # Truncation startegy for max_length (not longest)
            if truncation and len(sa_list) > max_length: 
                sa_list = sa_list[:max_length]
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
        self.token2id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.unk_idx = self.token2id[self.unk_token]
        self.pad_idx = self.token2id[self.pad_token]
        self.cls_idx = self.token2id[self.cls_token]
        self.mask_idx = self.token2id[self.mask_token]
        self.eos_idx = self.token2id[self.eos_token]

    def __call__(self, pdb_list,
                 truncation=True,
                 max_length=max_length,
                 return_tensors='pt'):

        input_ids = []
        attention_masks = []

        seqs = [self.extract_aa_seq(pdb) for pdb in pdbs]
        for seq in seqs:
            print(seq)
            print(len(seq))
            print('---------------------')

    def extract_aa_seq(self, pdb_path, chain_id='A'): 
        with open(pdb_path, 'r') as pdb_file:
            for record in SeqIO.parse(pdb_file, 'pdb-atom'):
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
        self.token2id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.unk_idx = self.token2id[self.unk_token]
        self.pad_idx = self.token2id[self.pad_token]
        self.cls_idx = self.token2id[self.cls_token]
        self.mask_idx = self.token2id[self.mask_token]
        self.eos_idx = self.token2id[self.eos_token]

    def __call__(self):
        pass

if __name__ == "__main__":
    import glob

    structures_directory = 'structures/'
    pdbs = glob.glob('%s*.pdb'%structures_directory)
    tokenizer = SequenceTokenizer()
    inputs = tokenizer(pdbs)
    print(inputs)
