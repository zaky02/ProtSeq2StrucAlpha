import itertools
from SaProt.utils.foldseek_util import get_struc_seq

foldseek_seq_vocab = "ACDEFGHIKLMNPQRSTVWY#"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
max_length = 1024

foldseek_path = '/home/phastos/Programs/mambaforge/envs/SaProt/lib/python3.10/site-packages/SaProt/bin/foldseek'

class SaProtTokenizer:
    def __init__(self):
        self.cls_token = '<cls>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        # It is being use? How so? There are many different ways of masking
        self.mask_token = '<mask>'
        self.build_vocab()

    def build_vocab(self):
        self.tokens = [self.cls_token,
                       self.pad_token,
                       self.eos_token,
                       self.unk_token,
                       self.mask_token]

        for seq_token, struc_token in itertools.product(foldseek_seq_vocab,
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

        for pdb in pdb_list:
            seq, foldseek, combined = self.get_sa_vocab(pdb)

        if truncation and len(combined) > max_length -2: 
        # max_length -2 to take into account the cls and eos tokens
            combined = combined[:max_length - 2]

        combined = list(combined)
        combined = [self.cls_token] + combined + [self.eos_token]
        print(combined)
            
        
    def get_sa_vocab(self, pdb_path, chain_id='A'):
        parsed_seqs = get_struc_seq(foldseek_path, pdb_path)[chain_id]
        seq, foldseek,  combined = parsed_seqs
        return seq, foldseek, combined

if __name__ == "__main__":
    import glob

    structures_directory = 'structures/'
    pdbs = glob.glob('%s*.pdb'%structures_directory)
    tokenizer = SaProtTokenizer()
    tokenizer(pdbs)
