from pathlib import Path

import numpy as np
import os
import multiprocessing
from argparse import ArgumentParser



ROOT_DIR = "/usace_share/SewerML"





def sample(list_files, sample_size):
    sampleIndexes = np.random.uniform(0, len(list_files), sample_size).astype(int)
    return list_files[sampleIndexes]





def create_symbolic_link(source_file, destination_dir):
    source_path = os.path.abspath(source_file)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    try:
        os.symlink(source_path, destination_path)
        # print(f"[+] Symbolic link created: {source_path} -> {destination_path}")
    except FileExistsError:
        print(f"[-] Symbolic link already exists: {destination_path}")
    except OSError as e:
        print(f"[-] Error creating symbolic link: {e}")

def link_files(source_files, destination_dir):


    # Create a multiprocessing.Pool with the number of processes you want to use
    pool = multiprocessing.Pool()

    # Use the pool to create symbolic links in parallel
    for file in source_files:
        pool.apply_async(create_symbolic_link, args=(file, destination_dir))

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1325)
    parser.add_argument('--folder', type= str, default="train00" )
    parser.add_argument('--outdir', type=str, default="train")
    parser.add_argument('--num-samples', type=int, default=100)




    args = parser.parse_args()

    folderName = args.folder
    np.random.seed(args.seed)
    OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    folderDir = os.path.join(ROOT_DIR, folderName)
    list_files = np.array([img_file for img_file in Path(folderDir).glob('*.png')])
    sample_files = sample(list_files, args.num_samples)

    link_files(sample_files, OUTDIR)

