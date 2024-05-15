import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='Generate train, validation splits for the dataset.')
parser.add_argument('--input_dir', type=str, help='Directory containing the images')
parser.add_argument('--output_dir', type=str, help='Path to save the train, val, test splits.')
args = parser.parse_args()

def get_ids(image_path):
    """
    Find all file ids in the image directory
    """
    ids = []
    for file in os.listdir(image_path):
        if file.endswith(".jpg"):
            file_id = file.split(".")[0]
            if file_id not in ids:
                ids.append(file_id)
    return ids

def split_train_val(ids, train_ratio=0.8, val_ratio=0.2, output_dir=None):
    """
    Split the IDs into train and validation sets
    """
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio == 1, "The sum of splits ratios must be 1"

    # List all image files in the data directory
    id_list = ids

    # Shuffle the list of IDs to ensure random distribution
    np.random.shuffle(id_list)

    # Split the IDs based on the given ratios
    total_images = len(id_list)
    train_end = int(total_images * train_ratio)

    train_ids = id_list[:train_end]
    val_ids = id_list[train_end:]

    # Delete existing files
    if os.path.exists('train_ids.txt'):
        os.remove('train_ids.txt')
    if os.path.exists('val_ids.txt'):
        os.remove('val_ids.txt')

    # Save IDs as files in the output directory
    if output_dir:
        train_file = os.path.join(output_dir, 'train_ids.txt')
        val_file = os.path.join(output_dir, 'val_ids.txt')
    else:
        train_file = 'train_ids.txt'
        val_file = 'val_ids.txt'

    with open(train_file, 'w') as f:
        for id in train_ids:
            f.write("%s\n" % id)

    with open(val_file, 'w') as f:
        for id in val_ids:
            f.write("%s\n" % id)

def check_duplicates(id_dir):
    """
    Check for duplicate IDs in the given directory (3 files)
    :param id_dir:
    :return:
    """
    unique_ids = []
    for file in os.listdir(id_dir):
        if file.endswith(".txt"):
            with open(file, 'r') as f:
                for id in f:
                    if id in unique_ids:
                        print(f"Duplicate ID: {id}")
                    else:
                        unique_ids.append(id)
    print("No duplicates found.")



if __name__ == '__main__':
    # Path to the image directory
    path = args.input_dir

    # Get all image IDs
    ids = get_ids(path)

    split_train_val(ids, output_dir=args.output_dir)

    # Check for duplicates
    check_duplicates(args.output_dir)
