
import argparse
import os

import numpy as np



parser = argparse.ArgumentParser(description='For each id in the given .txt file, get the emotion count')
parser.add_argument('--id_file', type=str, help='Path to the .txt file containing the ids')
parser.add_argument('--annotations_dir', type=str, help='Path to the directory containing the annotations')
args = parser.parse_args()

def get_emotions(annotation_path, id_file):
    """
    Get the emotions for ids in the given .txt file
    """
    emotion_count = {}
    file_path = os.path.join(annotation_path, id_file)
    with open(file_path, "r") as f:
        for id in f:
            id = id.strip()
            emotion = np.load(f"{annotation_path}/{id}_exp.npy")
            # Turn emotion into int
            emotion = int(emotion)
            if emotion not in emotion_count:
                emotion_count[emotion] = 1
            else:
                emotion_count[emotion] += 1
    return emotion_count


if __name__ == "__main__":
    emotion_count = get_emotions(args.annotations_dir, args.id_file)
    print(emotion_count)
