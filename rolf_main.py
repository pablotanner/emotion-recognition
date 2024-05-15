import argparse
import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from src.data_processing.rolf_loader import RolfLoader

parser = argparse.ArgumentParser(description='Model training and evaluation.')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)')
args = parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading data...")
    data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir, args.test_features_dir, args.main_id_dir)
    logger.info("Data loaded.")

    features_to_use = ['nonrigid_face_shape', 'landmarks_3d', 'facs_intensity', 'hog']

    features, emotions = data_loader.get_data()

    logger.info("Preprocessing data...")
    # First apply PCA on HOG
    pca = PCA(n_components=0.95)
    features['train']['hog'] = pca.fit_transform(features['train']['hog'])
    features['val']['hog'] = pca.transform(features['val']['hog'])
    features['test']['hog'] = pca.transform(features['test']['hog'])

    # Use MinMax Scaler for all
    for feature in features_to_use:
        scaler = MinMaxScaler(feature_range=(-5, 5))
        features['train'][feature] = scaler.fit_transform(features['train'][feature])
        features['val'][feature] = scaler.transform(features['val'][feature])
        features['test'][feature] = scaler.transform(features['test'][feature])

    logger.info("Data preprocessed.")


    logger.info("Concatenating features and splitting data...")
    X_train = []
    X_val = []
    X_test = []

    for feature in features_to_use:
        X_train.append(features['train'][feature])
        X_val.append(features['val'][feature])
        X_test.append(features['test'][feature])

    X_train = np.concatenate(X_train, axis=1)
    X_val = np.concatenate(X_val, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    y_train = emotions['train']
    y_val = emotions['val']
    y_test = emotions['test']


    logger.info("Data concatenated and split.")

    # Print shapes
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_val shape: {y_val.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    # Save the data
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)

    logger.info("Data saved.")


