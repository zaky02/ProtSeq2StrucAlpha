# Script based on SaProt/utils/foldseek_util.py from https://github.com/SaProt/SaProt

import os


def get_foldseek_seq(foldseek,
                     pdb,
                     chains: list = None,
                     process_id: int = 0) -> dict:
    """
    Args:
        foldseek: Binary executable file of foldseek
        pdb: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains
                will be extracted.
        process_id: Process ID for temporary files. This is used for parallel
                    processing.

    Returns:
        struc_seq_dict: A dict of structural sequences. The keys are chain IDs
                        and the values are the structural sequences.
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(pdb), f"Pdb file not found: {pdb}"
    
    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {pdb} {tmp_save_path}"
    os.system(cmd)

    struc_seq_dict = {}
    name = os.path.basename(pdb)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            struc_seq = "".join([b.lower() for b in list(struc_seq)])
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in struc_seq_dict:
                    struc_seq_dict[chain] = (seq, struc_seq)
        
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    
    return struc_seq_dict
