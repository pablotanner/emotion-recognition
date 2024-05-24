import argparse
import json
import logging
import os
import numpy as np
from cuml.svm import LinearSVC, SVR
from cuml.ensemble import RandomForestClassifier as RFC
from cuml.neighbors import KNeighborsClassifier as KNN
from cuml.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import make_scorer, balanced_accuracy_score
from torch import optim
from src.model_training.torch_mlp import PyTorchMLPClassifier
from src.model_training.torch_neural_network import NeuralNetwork
from sklearn.model_selection import ParameterGrid
from joblib import parallel_backend

def save_checkpoint(grid_search_state, filename):
    with open(filename, 'w') as f:
        json.dump(grid_search_state, f)

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

# Experiments for finding optimal classifier for the different features
parser = argparse.ArgumentParser(description='Finding optimal classifier for different features')
parser.add_argument('--feature', type=str, help='Feature to use', default='landmarks_3d')
parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file', default='/local/scratch/ptanner/individual_experiments')
parser.add_argument('--dummy', action='store_true', help='Use dummy data')
args = parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'individual_logs/{args.feature}.log'),
                            logging.StreamHandler()
                        ])

    logger.info(f'Running experiments for feature {args.feature}')

    parameters = {
        'SVM': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale','auto']},
        'LinearSVC': {'C': [0.1, 1, 10, 100]},
        'RandomForest': {'n_estimators': [100, 200, 300, 400], 'max_depth': [10, 15, 20], 'min_samples_split': [2, 4], 'split_criterion': [0,1]},
        'KNN': {'n_neighbors': [3, 5, 7, 9]},
        'LogisticRegression': {'C': [0.1, 1, 10, 100]},
        'MLP': {'hidden_size': [64, 128, 256], 'num_epochs': [10, 20, 30], 'batch_size': [32, 64, 128], 'learning_rate': [0.001, 0.01, 0.1]},
        'NN':  {'num_epochs': [10, 20, 30], 'batch_size': [32, 64, 128]}
    }

    feature_files = {
        'landmarks_3d': ['train_spatial_features.npy', 'val_spatial_features.npy', 'test_spatial_features.npy'],
        'embedded': ['train_embedded_features.npy', 'val_embedded_features.npy', 'test_embedded_features.npy'],
        'facs': ['train_facs_features_features.npy', 'val_facs_features.npy', 'test_facs_features.npy'],
        'pdm': ['train_pdm_features.npy', 'val_pdm_features.npy', 'test_pdm_features.npy'],
        'hog': ['pca_train_hog_features.npy', 'pca_val_hog_features.npy', 'pca_test_hog_features.npy'],
    }

    ros = RandomOverSampler(random_state=0)

    logger.info('Loading and Resampling data')

    if args.dummy:
        X_train, y_train = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
        X_val, y_val = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
        X_test, y_test = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
    else:
        X_train, y_train = ros.fit_resample(np.load(feature_files[args.feature][0]), np.load('y_train.npy'))
        X_val, y_val = ros.fit_resample(np.load(feature_files[args.feature][1]), np.load('y_val.npy'))
        X_test, y_test = ros.fit_resample(np.load(feature_files[args.feature][2]), np.load('y_test.npy'))


    classifiers = {
        'LinearSVC': LinearSVC,
        'SVC': SVR,
        'RandomForest': RFC,
        'KNN': KNN,
        'LogisticRegression': LogisticRegression,
        'MLP': PyTorchMLPClassifier(input_size=X_train.shape[1], num_classes=len(np.unique(y_train))),
        'NN': NeuralNetwork(input_dim=X_train.shape[1]),
    }


    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

    checkpoint_file = f'{args.experiment_dir}/checkpoints/{args.feature}.json'
    grid_search_state = load_checkpoint(checkpoint_file)


    best_classifiers = {}


    for clf_name, clf in classifiers.items():
        try:
            logger.info(f'Running experiments for classifier {clf_name}')
            param_grid = list(ParameterGrid(parameters[clf_name]))

            if clf_name not in grid_search_state:
                grid_search_state[clf_name] = {'best_score': 0, 'best_params': None, 'tried_params': []}

            best_score = grid_search_state[clf_name]['best_score']
            best_params = grid_search_state[clf_name]['best_params']
            tried_params = grid_search_state[clf_name]['tried_params']

            with parallel_backend('threading'):
                for params in param_grid:
                    if params in tried_params:
                        continue

                clf.set_params(**params)

                logger.info(f'Fitting model with parameters {params}')
                if clf_name == 'NN':
                    clf.compile(optim.Adam(clf.parameters(), lr=0.001))
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                score = balanced_accuracy_score(y_val, y_val_pred)

                if score > best_score:
                    best_score = score
                    best_params = params

                # Update the checkpoint state
                grid_search_state[clf_name]['best_score'] = best_score
                grid_search_state[clf_name]['best_params'] = best_params
                grid_search_state[clf_name]['tried_params'].append(params)
                save_checkpoint(grid_search_state, checkpoint_file)

            best_classifiers[clf_name] = clf.set_params(**best_params)
            best_classifiers[clf_name].fit(X_train, y_train)
            logger.info(f'Best parameters for {clf_name}: {best_params}')

            y_pred = best_classifiers[clf_name].predict(X_val)
            logger.info(f'Validation score for {clf_name}: {balanced_accuracy_score(y_val, y_pred)}')

        except Exception as e:
            logger.error(f'Error while fitting model: {e}')
            continue

    for clf_name, best_clf in best_classifiers.items():
        y_pred = best_clf.predict(X_test)
        logger.info(f'Test score for {clf_name}: {balanced_accuracy_score(y_test, y_pred)}')










