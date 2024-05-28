import argparse
import gc
import logging
import os
import numpy as np
#from cuml.decomposition import IncrementalPCA as PCA
#from cuml.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import compute_class_weight
from cuml.svm import SVC
from src import evaluate_results
from src.data_processing.rolf_loader import RolfLoader
from src.model_training.torch_mlp import PyTorchMLPClassifier
from src.model_training.torch_neural_network import NeuralNetwork

import cupy as cp

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
parser.add_argument('--load-gpu', action='store_true', help='Load data to GPU')
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

    path = f'{args.experiment_dir}/{dataset_type}_concatenated_features.npy'

    if os.path.exists(path):
        return path

    X_list = []

    for feature in feature_types.keys():
        logger.info(f'Loading {feature}...')

        # Use memory mapping to load data
        file_path = f'{args.experiment_dir}/{dataset_type}_{feature}.npy'

        if args.load_gpu:
            data = cp.load(file_path, mmap_mode='r').astype(np.float32)
        else:
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

    np.save(path, X)

    logger.info('Data saved')

    return path




def preprocess_and_save_features(X_train, X_val, X_test, feature_name, use_minmax=False):
    # Check if the feature is already preprocessed
    if f'train_{feature_name}.npy' in os.listdir(args.experiment_dir):
        logger.info(f'{feature_name} already preprocessed')
        return

    logger.info(f'Scaling {feature_name}...')
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)

    # Initialize Scaler
    if use_minmax:
        scaler = MinMaxScaler(feature_range=(-5, 5))
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    logger.info(f'Dimensionality Reduction for {feature_name}...')

    # Step 2: Dimensionality Reduction
    if X_train.shape[1] < 50:
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)


    # Save the preprocessed features
    np.save(f'{args.experiment_dir}/train_{feature_name}.npy', X_train)
    np.save(f'{args.experiment_dir}/val_{feature_name}.npy', X_val)
    np.save(f'{args.experiment_dir}/test_{feature_name}.npy', X_test)

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
            logger.info('USING DUMMY DATA')
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
            logger.info(f'Loading Data with RolfLoader')

            data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                     args.test_features_dir, args.main_id_dir)
            feature_splits_dict, emotions_splits_dict = data_loader.get_data()


        # Save the emotion labels
        y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']
        np.save(f'{args.experiment_dir}/y_train.npy', y_train)
        np.save(f'{args.experiment_dir}/y_val.npy', y_val)
        np.save(f'{args.experiment_dir}/y_test.npy', y_test)

        logger.info(f'Preprocessing Data and Saving...')
        for feature_name, linearity in feature_types.items():
            use_mm = feature_name in ['landmarks_3d', 'hog', 'sface', 'facenet','facs_intensity']

            preprocess_and_save_features(
                feature_splits_dict['train'][feature_name],
                feature_splits_dict['val'][feature_name],
                feature_splits_dict['test'][feature_name],
                feature_name,
                use_minmax=use_mm,
            )
    gc.collect()
    logger.info(f'Preparing concatenated data')
    y_train = np.load(f'{args.experiment_dir}/y_train.npy')

    X_train_path = load_and_concatenate_features('train')
    X_val_path = load_and_concatenate_features('val')
    X_test_path = load_and_concatenate_features('test')


    X_train = np.load(X_train_path)

    logger.info(f'Prepared/Loaded concat data...')
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    nn = NeuralNetwork(input_dim=X_train.shape[1],  class_weight=class_weights, num_epochs=10, batch_size=128)
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights)
    svm = SVC(class_weight=class_weights, probability=True, kernel='rbf', C=1)
    nn.__class__.__name__ = 'NeuralNetwork'
    rf.__class__.__name__ = 'RandomForestClassifier'
    svm.__class__.__name__ = 'SVC'

    models = [
        nn,
        svm,
        rf,
    ]

    probabilities_val = {}


    y_val = np.load(f'{args.experiment_dir}/y_val.npy')
    X_val = np.load(X_val_path)

    for model in models:
        logger.info(f'Training {model.__class__.__name__}...')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        bal_acc_val = balanced_accuracy_score(y_val, np.argmax(proba, axis=1))
        logger.info(f'Balanced Accuracy of {model.__class__.__name__} (Validation Set): {bal_acc_val}')

        probabilities_val[model.__class__.__name__] = proba
    
    del X_train, y_train
    del X_val

    gc.collect()

    def evaluate_stacking(probabilities, y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([('log_reg', LogisticRegression(C=1, solver='liblinear', class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        logger.info(f"Accuracy of stacking classifier (Validation Set): {stacking_accuracy}")

        balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
        logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")

        # Return the stacking pipeline
        return stacking_pipeline

    # Use stacking
    stacking_pipeline = evaluate_results(probabilities_val, y_val)