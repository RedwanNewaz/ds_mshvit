import os
import multiprocessing
from argparse import ArgumentParser

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

def link_files(source_dir, destination_dir):
    source_files = os.listdir(source_dir)

    # Create a multiprocessing.Pool with the number of processes you want to use
    pool = multiprocessing.Pool()

    # Use the pool to create symbolic links in parallel
    for file in source_files:
        pool.apply_async(create_symbolic_link, args=(os.path.join(source_dir, file), destination_dir))

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src-dir', type=str, default='/usace_share/SewerML/valid00', help='directory of image files')
    parser.add_argument('--dest-dir', type=str, default='/home/aredwann/PyDev/SewerML-tracker/Resources/Dataset/SewerML/Data/Validation/val_data', help='output symbolic image files')
    args = parser.parse_args()

    link_files(args.src_dir, args.dest_dir)