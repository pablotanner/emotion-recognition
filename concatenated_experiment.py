import argparse
import gc
import logging
import os
import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
#from cuml.decomposition import IncrementalPCA as PCA
#from cuml.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import compute_class_weight

from src import evaluate_results
from src.data_processing.rolf_loader import RolfLoader
from src.model_training.torch_mlp import PyTorchMLPClassifier

# Experiments for optimizing EmoRec with concatenated feature approach (feature fusion)
parser = argparse.ArgumentParser(description='Optimizing EmoRec with feature fusion approach')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
parser.add_argument('--experiment-dir', type=str, help='Directory to experiment dir', default='/local/scratch/ptanner/concatenated_experiment')
parser.add_argument('--dummy', action='store_true', help='Use dummy data')
parser.add_argument('--skip-loading',  action='store_true', help='Skip preprocessing and loading data')
args = parser.parse_args()

feature_types = {
    'landmarks_3d': 'linear',
    'facs_intensity': 'linear',
    'facs_presence': 'linear',
    'hog': 'nonlinear',
    'facenet': 'nonlinear',
    'sface': 'nonlinear',
    'nonrigid_face_shape': 'nonlinear'
}

def load_and_concatenate_features(dataset_type):
    logger.info('Loading Data')

    X_list = []

    for feature in feature_types.keys():
        logger.info(f'Loading {feature}...')

        # Use memory mapping to load data
        file_path = f'{args.experiment_dir}/{dataset_type}_{feature}.npy'
        data = np.load(file_path, mmap_mode='r').astype(np.float32)
        X_list.append(data)

        # Explicitly free memory
        del data
        gc.collect()

    logger.info(f'Concatenating features for {dataset_type}...')
    X = np.concatenate(X_list, axis=1)

    # Free the list memory
    del X_list
    gc.collect()

    logger.info('Data loaded and concatenated')

    return X


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
    # Check if the feature is already preprocessed
    if f'train_{feature_name}.npy' in os.listdir(args.experiment_dir):
        logger.info(f'{feature_name} already preprocessed')
        return

    logger.info(f'Scaling {feature_name}...')
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    # Initialize Dask StandardScaler
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    logger.info(f'Dimensionality Reduction for {feature_name}...')

    # Step 2: Dimensionality Reduction
    if feature_type == 'linear' and X_train.shape[1] > 50:
        if n_components is None:
            n_components = min(X_train.shape[1], 50)
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
    elif feature_type == 'nonlinear' and X_train.shape[1] > 50:
        if autoencoder_components is None:
            autoencoder_components = min(X_train.shape[1], 50)
        encoder, X_train = fit_autoencoder(X_train, autoencoder_components)
        X_val = apply_autoencoder(X_val, encoder)
        X_test = apply_autoencoder(X_test, encoder)


    # Save the preprocessed features
    np.save(os.path.join(args.experiment_dir, f'train_{feature_name}.npy'), X_train)
    np.save(os.path.join(args.experiment_dir, f'val_{feature_name}.npy'), X_val)
    np.save(os.path.join(args.experiment_dir, f'test_{feature_name}.npy'), X_test)

    logger.info(f'{feature_name} preprocessing complete.')
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

    if not args.skip_loading:
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


        for feature_name, linearity in feature_types.items():
            if linearity == 'linear':
                preprocess_and_save_features(feature_splits_dict['train'][feature_name], feature_splits_dict['val'][feature_name], feature_splits_dict['test'][feature_name], feature_name, linearity, n_components=50)
            else:
                preprocess_and_save_features(feature_splits_dict['train'][feature_name], feature_splits_dict['val'][feature_name], feature_splits_dict['test'][feature_name], feature_name, linearity, autoencoder_components=50)

    gc.collect()
    logger.info(f'Loading Data')
    y_train = np.load(f'{args.experiment_dir}/y_train.npy')
    #y_val = np.load(f'{args.experiment_dir}/y_val.npy')
    #y_test = np.load(f'{args.experiment_dir}/y_test.npy')

    X_train = load_and_concatenate_features('train')
    #X_val = np.concatenate([np.load(f'{args.experiment_dir}/val_{feature}.npy').astype(np.float32) for feature in feature_types.keys()], axis=1)
    #X_test = np.concatenate([np.load(f'{args.experiment_dir}/test_{feature}.npy').astype(np.float32) for feature in feature_types.keys()], axis=1)


    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    mlp = PyTorchMLPClassifier(X_train.shape[1],num_classes=8, class_weight=class_weights, hidden_size=64, num_epochs=10, batch_size=64, learning_rate=0.01)

    logger.info(f'Fitting MLP')
    mlp.fit(X_train, y_train)
    del X_train, y_train
    
    y_val = np.load(f'{args.experiment_dir}/y_val.npy')

    X_val = load_and_concatenate_features('val')

    y_pred = mlp.predict(X_val)
    del X_val

    bal_acc = balanced_accuracy_score(y_val, y_pred)
    logger.info(f'Balanced Accuracy: {bal_acc}')
    evaluate_results(y_val, y_pred)

    logger.info('Experiment completed')