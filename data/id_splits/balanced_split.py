import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser(description='Generate train, validation splits for the dataset.')
parser.add_argument('--input_dir', type=str, help='Directory with annotation files')
parser.add_argument('--output_dir', type=str, help='Path to save the train, val, test splits as txts')
args = parser.parse_args()


def get_ids(annotation_path):
    """
    Find all file ids in the annotation directory
    """
    ids = []
    emotions = []
    for file in os.listdir(annotation_path):
        if file.endswith("_exp.npy"):
            file_id = file.split("_")[0]
            emotion = np.load(f"{annotation_path}/{file}").astype(int)
            if file_id not in ids:
                ids.append(file_id)
                emotions.append(emotion)
    return ids, emotions


def split_train_val(ids, emotions, train_ratio=0.8, val_ratio=0.2, output_dir=None):
    """
    Split the ids into train and val sets, where the val set is balanced
    """
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio == 1, "The sum of splits ratios must be 1"

    # Convert to numpy arrays
    ids = np.array(ids)
    emotions = np.array(emotions)

    # Stratified split to ensure balanced validation set
    train_ids, val_ids, train_emotions, val_emotions = train_test_split(
        ids, emotions, test_size=val_ratio, stratify=emotions, random_state=42
    )

    # Save IDs as files in the output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_file = os.path.join(output_dir, 'train_ids_bal.txt')
        val_file = os.path.join(output_dir, 'val_ids_bal.txt')
    else:
        train_file = 'train_ids_bal.txt'
        val_file = 'val_ids_bal.txt'

    with open(train_file, 'w') as f:
        for id in train_ids:
            f.write(f"{id}\n")

    with open(val_file, 'w') as f:
        for id in val_ids:
            f.write(f"{id}\n")

    return train_ids, train_emotions, val_ids, val_emotions

if __name__ == "__main__":
    data = get_ids(args.input_dir)
    split_train_val(data[0], data[1], output_dir=args.output_dir)
