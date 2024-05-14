import os

import numpy as np


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


def initialize_split_files(ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of splits ratios must be 1"

    # List all image files in the data directory
    id_list = ids

    # Shuffle the list of IDs to ensure random distribution
    np.random.shuffle(id_list)

    # Split the IDs based on the given ratios
    total_images = len(id_list)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    train_ids = id_list[:train_end]
    val_ids = id_list[train_end:val_end]
    test_ids = id_list[val_end:]

    # Delete existing files
    if os.path.exists('train_ids.txt'):
        os.remove('train_ids.txt')
    if os.path.exists('val_ids.txt'):
        os.remove('val_ids.txt')
    if os.path.exists('test_ids.txt'):
        os.remove('test_ids.txt')

    # Save IDs to files
    with open('train_ids.txt', 'w') as f:
        for id in train_ids:
            f.write("%s\n" % id)

    with open('val_ids.txt', 'w') as f:
        for id in val_ids:
            f.write("%s\n" % id)

    with open('test_ids.txt', 'w') as f:
        for id in test_ids:
            f.write("%s\n" % id)

    print(f"Initialized ID files: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test samples.")

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
    path = "../images/"

    # Get all image IDs
    ids = get_ids(path)


    #initialize_split_files(ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    check_duplicates('./')
