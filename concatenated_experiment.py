import argparse
import logging
import os

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from sklearn.decomposition import PCA
from cuml.preprocessing import StandardScaler, MinMaxScaler


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


def batch_generator(X_path, batch_size):
    X_data = np.load(X_path, mmap_mode='r')  # Use memory-mapped mode to avoid loading the whole array into memory
    n_samples = X_data.shape[0]
    indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        X_batch = X_data[batch_indices]
        yield X_batch

def partial_fit_scalers(standard_scaler, minmax_scaler, X_path, batch_size):
    for X_batch in batch_generator(X_path, batch_size):
        standard_scaler.partial_fit(X_batch)

    for X_batch in batch_generator(X_path, batch_size):
        X_batch_standard_scaled = standard_scaler.transform(X_batch)
        minmax_scaler.partial_fit(X_batch_standard_scaled)

def transform_in_batches(scaler, X_path, batch_size, output_path):
    X_data = np.load(X_path, mmap_mode='r')
    n_samples = X_data.shape[0]
    scaled_data = np.memmap(output_path, dtype='float32', mode='w+', shape=X_data.shape)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X_data[start_idx:end_idx]
        scaled_data[start_idx:end_idx] = scaler.transform(X_batch)

    return scaled_data

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

def preprocess_and_save_features(X_train_path, X_val_path, X_test_path, feature_name, feature_type, n_components=None, autoencoder_components=None, use_minmax=False, batch_size=2000):
    # Check if the feature is already preprocessed
    if f'train_{feature_name}.npy' in os.listdir(args.experiment_dir):
        logger.info(f'{feature_name} already preprocessed')
        return

    logger.info(f'Scaling {feature_name}...')
    # Step 1: Scaling
    if use_minmax:
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler(feature_range=(-5, 5))

        # Perform partial_fit on the training data
        partial_fit_scalers(standard_scaler, minmax_scaler, X_train_path, batch_size)

        # Transform the data in batches
        scaled_train_path = f'{args.experiment_dir}/scaled_train_{feature_name}.npy'
        scaled_val_path = f'{args.experiment_dir}/scaled_val_{feature_name}.npy'
        scaled_test_path = f'{args.experiment_dir}/scaled_test_{feature_name}.npy'

        transform_in_batches(standard_scaler, X_train_path, batch_size, scaled_train_path)
        transform_in_batches(standard_scaler, X_val_path, batch_size, scaled_val_path)
        transform_in_batches(standard_scaler, X_test_path, batch_size, scaled_test_path)

        X_train = np.load(scaled_train_path, mmap_mode='r')
        X_val = np.load(scaled_val_path, mmap_mode='r')
        X_test = np.load(scaled_test_path, mmap_mode='r')

    else:
        standard_scaler = StandardScaler()

        # Perform partial_fit on the training data
        for X_batch in batch_generator(X_train_path, batch_size):
            standard_scaler.partial_fit(X_batch)

        # Transform the data in batches
        scaled_train_path = f'{args.experiment_dir}/scaled_train_{feature_name}.npy'
        scaled_val_path = f'{args.experiment_dir}/scaled_val_{feature_name}.npy'
        scaled_test_path = f'{args.experiment_dir}/scaled_test_{feature_name}.npy'

        transform_in_batches(standard_scaler, X_train_path, batch_size, scaled_train_path)
        transform_in_batches(standard_scaler, X_val_path, batch_size, scaled_val_path)
        transform_in_batches(standard_scaler, X_test_path, batch_size, scaled_test_path)

        X_train = np.load(scaled_train_path, mmap_mode='r')
        X_val = np.load(scaled_val_path, mmap_mode='r')
        X_test = np.load(scaled_test_path, mmap_mode='r')

    logger.info(f'Dimensionality Reduction for {feature_name}...')

    logger.info(f'Dimensionality Reduction for {feature_name}...')
    # Step 2: Dimensionality Reduction
    if feature_type == 'linear' and X_train.shape[1] > 50:
        if n_components is None:
            n_components = min(X_train.shape[1], 50)
        pca, X_train_reduced = fit_pca(X_train, n_components)
        X_val_reduced = apply_pca(X_val, pca)
        X_test_reduced = apply_pca(X_test, pca)
    elif feature_type == 'nonlinear' and X_train.shape[1] > 50:
        if autoencoder_components is None:
            autoencoder_components = min(X_train.shape[1], 50)
        encoder, X_train_reduced = fit_autoencoder(X_train, autoencoder_components)
        X_val_reduced = apply_autoencoder(X_val, encoder)
        X_test_reduced = apply_autoencoder(X_test, encoder)
    else:
        X_train_reduced, X_val_reduced, X_test_reduced = X_train, X_val, X_test

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


    feature_types = {
        'landmarks_3d': 'linear',
        'facs_intensity': 'linear',
        'facs_presence': 'linear',
        'hog': 'nonlinear',
        'facenet': 'nonlinear',
        'sface': 'nonlinear',
        'nonrigid_face_shape': 'nonlinear'
    }

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
        if not os.path.exists(f'{args.experiment_dir}/unprocessed/train_landmarks_3d.npy'):
            data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                     args.test_features_dir, args.main_id_dir)
            feature_splits_dict, emotions_splits_dict = data_loader.get_data()

            # Save the emotion labels
            y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict[
                'test']
            np.save(f'{args.experiment_dir}/y_train.npy', y_train)
            np.save(f'{args.experiment_dir}/y_val.npy', y_val)
            np.save(f'{args.experiment_dir}/y_test.npy', y_test)

            for feature in feature_types.keys():
                np.save(f'{args.experiment_dir}/unprocessed/train_{feature}.npy', feature_splits_dict['train'][feature])
                np.save(f'{args.experiment_dir}/unprocessed/val_{feature}.npy', feature_splits_dict['val'][feature])
                np.save(f'{args.experiment_dir}/unprocessed/test_{feature}.npy', feature_splits_dict['test'][feature])


    for feature_name, linearity in feature_types.items():
        # Paths
        X_train_path = f'{args.experiment_dir}/unprocessed/train_{feature_name}.npy'
        X_val_path = f'{args.experiment_dir}/unprocessed/val_{feature_name}.npy'
        X_test_path = f'{args.experiment_dir}/unprocessed/test_{feature_name}.npy'
        if linearity == 'linear':
            #preprocess_and_save_features(feature_splits_dict['train'][feature_name], feature_splits_dict['val'][feature_name], feature_splits_dict['test'][feature_name], feature_name, linearity, n_components=50)
            preprocess_and_save_features(X_train_path, X_val_path, X_test_path, feature_name, linearity, n_components=50, use_minmax=True)
        else:
            #preprocess_and_save_features(feature_splits_dict['train'][feature_name], feature_splits_dict['val'][feature_name], feature_splits_dict['test'][feature_name], feature_name, linearity, autoencoder_components=50)
            preprocess_and_save_features(X_train_path, X_val_path, X_test_path, feature_name, linearity, autoencoder_components=50, use_minmax=True)
        logger.info(f'Preprocessed and saved {feature_name}')

    logger.info('Experiment completed')