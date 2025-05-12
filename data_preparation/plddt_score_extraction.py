import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from Bio import PDB

def plldt_score_extracter(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    residue_names = []
    plddt_scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    residue_name = f"{residue.get_resname()}_{residue.get_id()[1]}"
                    residue_names.append(residue_name)
                    plddt_scores.append(atom.bfactor)  # B-factor stores the pLDDT score
    
    csv_data = pd.DataFrame({'Residue': residue_names, 'pLDDT_score': plddt_scores})
    return csv_data
    
def plddt_plot(csv_data, pdb_file):
    x = np.arange(len(csv_data['pLDDT_score']))
    y = np.array(csv_data['pLDDT Score'].tolist())
    
    # Create line segments and apply colormap
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]])
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = plt.Normalize(vmin=0, vmax=100)
    
    # Plot with color gradient
    lc = LineCollection(segments, cmap='coolwarm_r', norm=norm, linewidth=2)
    lc.set_array(y)
    ax.add_collection(lc)
    
    # Add plot settings
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('pLDDT Score')
    plt.colorbar(lc, ax=ax, label='pLDDT Score')
    plt.title('pLDDT Score Plot with Smooth Color Gradient')
    plt.savefig("pdb_file[:-4]}.pdf§")


