import argparse
import json
import logging
import os
from cuml.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from src.thundersvm import *
from cuml.ensemble import RandomForestClassifier as RFC
from cuml.neighbors import KNeighborsClassifier as KNN
from cuml.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import make_scorer, balanced_accuracy_score
from torch import optim
from src.model_training.torch_mlp import PyTorchMLPClassifier
from src.model_training.torch_neural_network import NeuralNetwork
from sklearn.model_selection import ParameterGrid
import gc

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
parser.add_argument('--reset', action='store_true', help='Reset the checkpoints/logs')
parser.add_argument('--gpu-id', type=int, help='GPU ID to use', default=0)
args = parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    if args.reset:
        try:
            os.remove(f'{args.experiment_dir}/checkpoints/{args.feature}.json')
            os.remove(f'{args.experiment_dir}/logs/{args.feature}.log')
        except:
            pass

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/logs/{args.feature}.log'),
                            logging.StreamHandler()
                        ])

    logger.info(f'Running experiments for feature {args.feature}')

    if not os.path.exists(f'{args.experiment_dir}/{args.feature}'):
        os.makedirs(f'{args.experiment_dir}/{args.feature}')

    parameters = {
        'SVC': {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'polynomial'], 'gpu_id': [args.gpu_id]},
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
        'facs': ['train_facs_features.npy', 'val_facs_features.npy', 'test_facs_features.npy'],
        'pdm': ['train_pdm_features.npy', 'val_pdm_features.npy', 'test_pdm_features.npy'],
        'hog': ['pca_train_hog_features.npy', 'pca_val_hog_features.npy', 'pca_test_hog_features.npy'],
    }

    ros = RandomOverSampler(random_state=0)

    logger.info('Loading and Resampling data')

    X_shape = None

    if args.dummy:
        X_train, y_train = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
        X_val, y_val = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
        X_test, y_test = ros.fit_resample(np.random.rand(100, 10), np.random.randint(0, 2, 100))
    elif os.path.exists(f'{args.experiment_dir}/{args.feature}/X_train'):
        X_train = np.load(f'{args.experiment_dir}/{args.feature}/X_train.npy')
        y_train = np.load(f'{args.experiment_dir}/{args.feature}/y_train.npy')
        X_val = np.load(f'{args.experiment_dir}/{args.feature}/X_val.npy')
        y_val = np.load(f'{args.experiment_dir}/{args.feature}/y_val.npy')
        X_test = np.load(f'{args.experiment_dir}/{args.feature}/X_test.npy')
        y_test = np.load(f'{args.experiment_dir}/{args.feature}/y_test.npy')
    else:
        scaler = StandardScaler()
        X_train, y_train = ros.fit_resample(np.load(feature_files[args.feature][0]), np.load('y_train.npy'))
        X_val, y_val = ros.fit_resample(np.load(feature_files[args.feature][1]), np.load('y_val.npy'))
        X_test, y_test = ros.fit_resample(np.load(feature_files[args.feature][2]), np.load('y_test.npy'))

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        X_shape = X_train.shape[1]

        np.save(f'{args.experiment_dir}/{args.feature}/X_train.npy', X_train)
        np.save(f'{args.experiment_dir}/{args.feature}/y_train.npy', y_train)
        np.save(f'{args.experiment_dir}/{args.feature}/X_val.npy', X_val)
        np.save(f'{args.experiment_dir}/{args.feature}/y_val.npy', y_val)
        np.save(f'{args.experiment_dir}/{args.feature}/X_test.npy', X_test)
        np.save(f'{args.experiment_dir}/{args.feature}/y_test.npy', y_test)

    del ros, scaler, X_train, X_val, X_test
    gc.collect()

    logger.info('Data loaded and resampled')


    classifiers = {
        'LinearSVC': LinearSVC,
        #'SVC': SVC,
        'RandomForest': RFC,
        'KNN': KNN,
        'LogisticRegression': LogisticRegression,
        'MLP': PyTorchMLPClassifier,
        'NN': NeuralNetwork
    }


    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

    checkpoint_file = f'{args.experiment_dir}/checkpoints/{args.feature}.json'
    grid_search_state = load_checkpoint(checkpoint_file)


    best_classifiers = {}


    for clf_name, clf_class in classifiers.items():
        logger.info(f'Running experiments for classifier {clf_name}')
        param_grid = list(ParameterGrid(parameters[clf_name]))

        if clf_name not in grid_search_state:
            grid_search_state[clf_name] = {'best_score': 0, 'best_params': None, 'tried_params': []}

        best_score = grid_search_state[clf_name]['best_score']
        best_params = grid_search_state[clf_name]['best_params']
        tried_params = grid_search_state[clf_name]['tried_params']


        for params in param_grid:
            if params in tried_params:
                continue
            if clf_name == 'NN':
                clf = NeuralNetwork(input_dim=X_shape, **params )
                clf.compile(optim.Adam(clf.parameters(), lr=0.001))
            elif clf_name == 'MLP':
                clf = PyTorchMLPClassifier(input_size=X_shape, num_classes=len(np.unique(y_train)),
                                           **params)
            else:
                clf = clf_class(**params)

            logger.info(f'Fitting model with parameters {params}')
            X_train = np.load(f'{args.experiment_dir}/{args.feature}/X_train.npy')
            clf.fit(X_train, y_train)
            del X_train
            gc.collect()

            X_val = np.load(f'{args.experiment_dir}/{args.feature}/X_val.npy')

            # If classifier is NN or MLP, we need to convert probabilities to class labels
            if clf_name in ['NN', 'MLP']:
                y_val_pred = np.argmax(clf.predict_proba(X_val), axis=1)
            else:
                y_val_pred = clf.predict(X_val)

            del X_val
            gc.collect()

            score = balanced_accuracy_score(y_val, y_val_pred)

            if score > best_score:
                best_score = score
                best_params = params

            # Update the checkpoint state
            grid_search_state[clf_name]['best_score'] = best_score
            grid_search_state[clf_name]['best_params'] = best_params
            grid_search_state[clf_name]['tried_params'].append(params)
            save_checkpoint(grid_search_state, checkpoint_file)

        if clf_name == 'NN':
            best_classifiers[clf_name] = NeuralNetwork(input_dim=X_shape, **best_params)
            best_classifiers[clf_name].compile(optim.Adam(best_classifiers[clf_name].parameters(), lr=0.001))
        elif clf_name == 'MLP':
            best_classifiers[clf_name] = PyTorchMLPClassifier(input_size=X_shape, num_classes=len(np.unique(y_train)),
                                           **best_params)
        else:
            best_classifiers[clf_name] = clf_class(**best_params)

        X_train = np.load(f'{args.experiment_dir}/{args.feature}/X_train.npy')
        best_classifiers[clf_name].fit(X_train, y_train)
        del X_train
        gc.collect()
        logger.info(f'Best parameters for {clf_name}: {best_params}')

        X_val = np.load(f'{args.experiment_dir}/{args.feature}/X_val.npy')
        y_pred = best_classifiers[clf_name].predict(X_val)
        del X_val
        gc.collect()
        logger.info(f'Validation score for {clf_name}: {balanced_accuracy_score(y_val, y_pred)}')


    for clf_name, best_clf in best_classifiers.items():
        y_pred = best_clf.predict(np.load(f'{args.experiment_dir}/{args.feature}/X_test.npy'))
        logger.info(f'Test score for {clf_name}: {balanced_accuracy_score(y_test, y_pred)}')










