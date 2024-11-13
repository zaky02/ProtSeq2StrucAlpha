import os
import glob
import shutil
from collections import Counter
from utils.foldseek import get_struc_seq

pdb_path = glob.glob("/home/phastos/Projects/data/isoul/ProtSeq2StrucAlpha/structures/mane_overlap_v4/*.pdb")

output_dir = "/home/phastos/Projects/data/isoul/ProtSeq2StrucAlpha/structures/filtered_structures"
os.makedirs(output_dir, exist_ok=True)

for pdb in pdb_path:
    raw_data = get_struc_seq("/home/phastos/Projects/data/isoul/ProtSeq2StrucAlpha/bin/foldseek",
                             pdb, chains=['A'])
    if 'A' in raw_data:
        aa_seq = raw_data['A'][0]
        struc_seq = raw_data['A'][1]
        
        # Filtering steps
        common_char, count = Counter(struc_seq).most_common(1)[0]
        
        if (count / len(struc_seq)) <= 0.95 and len(aa_seq) > 30:
            shutil.copy(pdb, output_dir)
        else:
            print("Filtered ", pdb, ":")
            print("Amino Acid sequence: ", aa_seq)
            print("3Di structural sequence: ", struc_seq)
            print("-------------------------------------")

