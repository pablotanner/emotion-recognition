import argparse
import logging
import os

import numpy as np
from keras import Input, Model
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_processing.rolf_loader import RolfLoader

# Experiments for optimizing EmoRec with concatenated feature approach (feature fusion)
parser = argparse.ArgumentParser(description='Optimizing EmoRec with feature fusion approach')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
parser.add_argument('--experiment-dir', type=str, help='Directory to experiment dir', default='/local/scratch/ptanner/concatenated_experiment')
parser.add_argument('--dummy', action='store_true', help='Use dummy data')
args = parser.parse_args()


def fit_scalers(X_train):
    standard_scaler = StandardScaler()
    X_train_standard_scaled = standard_scaler.fit_transform(X_train)

    minmax_scaler = MinMaxScaler(feature_range=(-5, 5))
    X_train_scaled = minmax_scaler.fit_transform(X_train_standard_scaled)

    return standard_scaler, minmax_scaler, X_train_scaled


def apply_scalers(X, standard_scaler, minmax_scaler):
    X_standard_scaled = standard_scaler.transform(X)
    return minmax_scaler.transform(X_standard_scaled)

def fit_pca(X_train_scaled, n_components):
    pca = PCA(n_components=n_components)
    X_train_reduced = pca.fit_transform(X_train_scaled)
    return pca, X_train_reduced

def apply_pca(X_scaled, pca):
    return pca.transform(X_scaled)

def fit_autoencoder(X_train_scaled, autoencoder_components):
    input_dim = X_train_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(autoencoder_components, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=256, shuffle=True, validation_split=0.1, callbacks=[early_stopping])

    X_train_reduced = encoder.predict(X_train_scaled)
    return encoder, X_train_reduced

def apply_autoencoder(X_scaled, encoder):
    return encoder.predict(X_scaled)

def preprocess_and_save_features(X_train, X_val, X_test, feature_name, feature_type, n_components=None, autoencoder_components=None, use_minmax=False):
    logger.info(f'Scaling {feature_name}...')
    # Step 1: Scaling
    if use_minmax:
        standard_scaler, minmax_scaler, X_train_scaled = fit_scalers(X_train)
        X_val_scaled = apply_scalers(X_val, standard_scaler, minmax_scaler)
        X_test_scaled = apply_scalers(X_test, standard_scaler, minmax_scaler)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

    logger.info(f'Dimensionality Reduction for {feature_name}...')
    # Step 2: Dimensionality Reduction
    if feature_type == 'linear':
        if n_components is None:
            n_components = min(X_train.shape[1], 50)
        pca, X_train_reduced = fit_pca(X_train_scaled, n_components)
        X_val_reduced = apply_pca(X_val_scaled, pca)
        X_test_reduced = apply_pca(X_test_scaled, pca)
    elif feature_type == 'nonlinear':
        if autoencoder_components is None:
            autoencoder_components = min(X_train.shape[1], 50)
        encoder, X_train_reduced = fit_autoencoder(X_train_scaled, autoencoder_components)
        X_val_reduced = apply_autoencoder(X_val_scaled, encoder)
        X_test_reduced = apply_autoencoder(X_test_scaled, encoder)
    else:
        X_train_reduced, X_val_reduced, X_test_reduced = X_train_scaled, X_val_scaled, X_test_scaled

    logger.info(f'Saving {feature_name}...')
    np.save(f'{args.experiment_dir}/train_{feature_name}.npy', X_train_reduced)
    np.save(f'{args.experiment_dir}/val_{feature_name}.npy', X_val_reduced)
    np.save(f'{args.experiment_dir}/test_{feature_name}.npy', X_test_reduced)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    experiment_name = input("Enter experiment name: ")


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/logs/{experiment_name}.log'),
                            logging.StreamHandler()
                        ])

    logger.info(f'Starting experiment')


    if args.dummy:
        num_samples = 1000

        feature_splits_dict = {
            'train': {
                'landmarks_3d': np.random.rand(num_samples, 68 * 3),
                'facs_intensity': np.random.rand(num_samples, 20),
                'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
                'nonrigid_face_shape': np.random.rand(num_samples, 13),
                'hog': np.random.rand(num_samples, 3000),
                'sface': np.random.rand(num_samples, 512),
                'facenet': np.random.rand(num_samples, 512)
            },
            'val': {
                'landmarks_3d': np.random.rand(num_samples, 68 * 3),
                'facs_intensity': np.random.rand(num_samples, 20),
                'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
                'nonrigid_face_shape': np.random.rand(num_samples, 13),
                'hog': np.random.rand(num_samples, 3000),
                'sface': np.random.rand(num_samples, 512),
                'facenet': np.random.rand(num_samples, 512)
            },
            'test': {
                'landmarks_3d': np.random.rand(num_samples, 68 * 3),
                'facs_intensity': np.random.rand(num_samples, 20),
                'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
                'nonrigid_face_shape': np.random.rand(num_samples, 13),
                'hog': np.random.rand(num_samples, 3000),
                'sface': np.random.rand(num_samples, 512),
                'facenet': np.random.rand(num_samples, 512)
            },
        }
        # 8 Classes
        emotions_splits_dict = {
            'train': np.random.randint(0, 8, num_samples),
            'val': np.random.randint(0, 8, num_samples),
            'test': np.random.randint(0, 8, num_samples)
        }
    else:
        data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                 args.test_features_dir, args.main_id_dir)
        feature_splits_dict, emotions_splits_dict = data_loader.get_data()


    # Save the emotion labels
    y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']
    np.save(f'{args.experiment_dir}/y_train.npy', y_train)
    np.save(f'{args.experiment_dir}/y_val.npy', y_val)
    np.save(f'{args.experiment_dir}/y_test.npy', y_test)


    feature_types = {
        'landmarks_3d': 'linear',
        'facs_intensity': 'linear',
        'facs_presence': 'linear',
        'hog': 'nonlinear',
        'facenet': 'nonlinear',
        'sface': 'nonlinear',
        'nonrigid_face_shape': 'nonlinear'
    }

    for feature_name, linearity in feature_types.items():
        preprocess_and_save_features(feature_splits_dict['train'][feature_name], feature_splits_dict['val'][feature_name], feature_splits_dict['test'][feature_name], feature_name, linearity)
        logger.info(f'Preprocessed and saved {feature_name}')

    logger.info('Experiment completed')