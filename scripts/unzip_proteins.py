import os
import gzip
import shutil


def unzip_pdb_gz_files(dir):
    # Check if the dir exists
    if not os.path.isdir(dir):
        print(f"dir '{dir}' does not exist.")
        return

    # Iterate over all files in the dir
    for filename in os.listdir(dir):
        if filename.endswith('.pdb.gz') or filename.endswith('.cif.gz'):
            gz_path = os.path.join(dir, filename)
            pdb_filename = filename[:-3]  # Remove the '.gz' extension
            pdb_path = os.path.join(dir, pdb_filename)

            # Check if the unzipped file already exists to avoid overwriting
            if os.path.exists(pdb_path):
                print(f"File '{pdb_path}' already exists. Skipping.")
                continue

            try:
                # Open the .gz file in binary read mode
                with gzip.open(gz_path, 'rb') as f_in:
                    # Open the target .pdb file in binary write mode
                    with open(pdb_path, 'wb') as f_out:
                        # Copy the contents from the .gz file to the .pdb file
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"Failed to unzip '{gz_path}': {e}")


if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Unzip all\
                                     .pdb.gz files in a directory.")
    parser.add_argument('--dir', type=str,
                        help='Path to the directory containing .pdb.gz files.')

    args = parser.parse_args()

    # Call the unzip function with the provided dir
    unzip_pdb_gz_files(args.dir)
