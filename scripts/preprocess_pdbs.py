import os
import sys
import glob
import csv
import shutil
from collections import Counter
from alive_progress import alive_bar
sys.path.append('..')
from utils import foldseek

def main(pdb_path, outname, filts):
    csv_path = os.path.dirname(outname)
    os.makedirs(csv_path, exist_ok=True)
    
    pdbs = glob.glob(pdb_path+'/*.pdb')

    with open(outname, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pdb", "chain", "aa_seq", "struc_seq"])
    
        with alive_bar(len(pdbs), bar="fish") as bar:
            for pdb in pdbs:
                raw_data = foldseek.get_struc_seq(
                    "bin/foldseek",
                    pdb)
                for chain in raw_data.keys():
                    aa_seq = raw_data[chain][0]
                    struc_seq = raw_data[chain][1]
                    
                    pdb_name = os.path.basename(pdb[:-4])

                    if filts:
                        common_char, count = Counter(struc_seq).most_common(1)[0]
                        if (count / len(struc_seq)) <= 0.90 and len(aa_seq) > 30:
                            writer.writerow([pdb_name, chain, aa_seq, struc_seq])
                        else:
                            print('Filtered PDB:%s chain:%s'%(pdb_name, chain))
                    else:
                        writer.writerow([pdb_name, chain, aa_seq, struc_seq])

                bar()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pdb_path',
                        help='directory with pdbs',
                        required=True)
    parser.add_argument('--outname',
                        help='',
                        required=True)
    parser.add_argument('--filts',
                        help='apply filters',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    pdb_path = args.pdb_path
    outname = args.outname
    filts = args.filts

    main(pdb_path=pdb_path, outname=outname, filts=filts)
