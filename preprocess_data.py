import os
import glob
import csv
import shutil
from collections import Counter
from utils.foldseek import get_struc_seq
from alive_progress import alive_bar

pdb_path = glob.glob("/home/phastos/Projects/data/ifilella/ProtSeq2StrucAlpha/structures/mane_overlap_v4/*.pdb")

output_dir = "/home/phastos/Projects/data/ifilella/ProtSeq2StrucAlpha/structures/filtered_structures"
os.makedirs(output_dir, exist_ok=True)

with alive_bar(bar='scuba') as bar:
    for pdb in pdb_path:
        raw_data = get_struc_seq("/home/phastos/Projects/data/ifilella/ProtSeq2StrucAlpha/bin/foldseek",
                                 pdb, chains=['A'])
        if 'A' in raw_data:
            aa_seq = raw_data['A'][0]
            struc_seq = raw_data['A'][1]
           
            pdb_suffix = os.path.basename(pdb[:-4])
            with open(f"{output_dir}/{pdb_suffix}.csv", "w") as file:
                writer = csv.writer(file)
                file.write("ID,aa_seq,struc_seq\n")
                file.write(f"{pdb_suffix},{aa_seq},{struc_seq}")
    
            # Filtering steps
            common_char, count = Counter(struc_seq).most_common(1)[0]
            
            if (count / len(struc_seq)) <= 0.95 and len(aa_seq) > 30:
                shutil.copy(pdb, output_dir)
            else:
                with open("filtered_proteins.txt", "a") as file:
                    print(f"Filtered {pdb}:", file=file)
                    print(f"Amino Acid sequence: {aa_seq}", file=file)
                    print(f"3Di structural sequence: {struc_seq}", file=file)
                    print("-------------------------------------", file=file)
            bar()

