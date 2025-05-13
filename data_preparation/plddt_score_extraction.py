import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Bio import PDB

def plldt_score_extracter(pdb_file):
   
    if os.path.getsize(pdb_file) == 0:
        print(f"[WARNING] Empty PDB file skipped: {pdb_file}")
        return 0.0 

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    plddt_scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    plddt_scores.append(atom.bfactor)  # B-factor stores the pLDDT score
    
    if not plddt_scores:
        print(f"[WARNING] No atoms found in file: {pdb_file}")
        return 0.0

    plddt = (np.mean(plddt_scores))/100
    print(f"The pLDDT score of the protein is: {plddt}")
    
    return plddt
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and plot pLDDT scores from PDB file")
    requiredArguments = parser.add_argument_group("Required Arguments")
    requiredArguments.add_argument('--pdb',
                                   help="PDB file to process",
                                   required=True)
    args = parser.parse_args()

    pdb = args.pdb

    plldt_score_extracter(pdb)
