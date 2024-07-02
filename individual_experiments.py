import argparse
import json
import logging
import os
import numpy as np
from cuml.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from src.model_training import SVC
#from cuml.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.utils import compute_class_weight
from torch import optim
from src.model_training.torch_mlp import PyTorchMLPClassifier
from src.model_training.torch_neural_network import NeuralNetwork
from sklearn.model_selection import ParameterGrid
import gc

from src.util.data_paths import get_data_path
from src.util.dataframe_converter import convert_to_cudf_df
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


    if args.feature == 'pdm':
        args.feature = 'nonrigid_face_shape'

    if not os.path.exists(f'{args.experiment_dir}/{args.feature}'):
        os.makedirs(f'{args.experiment_dir}/{args.feature}')

    scaler = StandardScaler()

    X_train = np.load(f'train_{args.feature}.npy').astype(np.float32)
    X_val = np.load(f'val_{args.feature}.npy').astype(np.float32)
    X_test = np.load(f'test_{args.feature}.npy').astype(np.float32)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    #X_train = np.load(get_data_path('train', args.feature)).astype(np.float32)
    #X_val = np.load(get_data_path('val', args.feature)).astype(np.float32)
    #X_test = np.load(get_data_path('test', args.feature)).astype(np.float32)
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    X_shape = X_test.shape[1]

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    dask_data = convert_to_cudf_df(X_train, X_val, X_test, y_train, y_val, y_test, npartitions=10)
    del X_train, X_val, X_test, y_train, y_val, y_test
    gc.collect()

    X_train = dask_data['X_train']
    y_train = dask_data['y_train']
    X_val = dask_data['X_val']
    y_val = dask_data['y_val']
    X_test = dask_data['X_test']
    y_test = dask_data['y_test']

    # Convert y to series
    y_train = y_train.iloc[:, 0]
    y_val = y_val.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    logger.info('Data loaded and resampled')


    parameters = {
        #'SVC': {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'probability': [True], 'class_weight':['balanced']},
        'SVC': {'C': [0.1, 1], 'kernel': ['rbf'], 'probability': [True], 'class_weight': ['balanced']},
        'LinearSVC': {'C': [0.1, 1, 10], 'class_weight':['balanced']},
        'RandomForest': {'n_estimators': [200, 300, 400], 'max_depth': [15, 20, None], 'min_samples_split': [2, 4], 'criterion': ['gini','entropy']},
        'LogisticRegression': {'C': [0.1, 1, 10, 100], 'class_weight':['balanced']},
        'MLP': {'hidden_size': [128, 256],'class_weight':[class_weights], 'num_epochs': [20, 30], 'batch_size': [64, 128], 'learning_rate': [0.001, 0.01]},
        'NN':  {'num_epochs': [10, 20, 30], 'batch_size': [64, 128], 'learning_rate':[0.001, 0.01], 'class_weight':[class_weights]}
    }


    classifiers = {
        'SVC': SVC,
        'LinearSVC': LinearSVC,
        'LogisticRegression': LogisticRegression,
        'MLP': PyTorchMLPClassifier,
        'NN': NeuralNetwork,
        'RandomForest': RandomForestClassifier,
    }


    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

    checkpoint_file = f'{args.experiment_dir}/checkpoints/{args.feature}.json'
    grid_search_state = load_checkpoint(checkpoint_file)


    best_classifiers = {}


    for clf_name, clf_class in classifiers.items():
        if clf_name != 'RandomForest':
            continue
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
                clf = NeuralNetwork(input_dim=X_shape, **params)
                clf.compile(optim.Adam(clf.parameters(), lr=0.001))
            elif clf_name == 'MLP':
                clf = PyTorchMLPClassifier(input_size=X_shape, num_classes=8,**params)
            else:
                clf = clf_class(**params)

            logger.info(f'Fitting model with parameters {params}')

            try:
                failed = False
                if clf_name in ['NN', 'MLP','LinearSVC']:
                    clf.fit(X_train.compute().to_numpy(), y_train.compute())
                elif clf_name in ['RandomForest']:
                    clf.fit(X_train.compute().to_numpy(), y_train.compute().to_numpy())
                else:
                    clf.fit(X_train.compute(), y_train.compute())
                #del X_train
                #gc.collect()

                # If classifier is NN or MLP, we need to convert probabilities to class labels
                if clf_name in ['NN', 'MLP']:
                    y_val_pred = np.argmax(clf.predict_proba(X_val.compute().to_numpy()), axis=1)
                    score = balanced_accuracy_score(y_val.compute().to_numpy(), y_val_pred)
                elif clf_name in ['RandomForest']:
                    y_val_pred = clf.predict(X_val.compute().to_numpy())
                    score = balanced_accuracy_score(y_val.compute().to_numpy(), y_val_pred)
                else:
                    y_val_pred = clf.predict(X_val.compute())
                    score = balanced_accuracy_score(y_val.compute().to_numpy(), y_val_pred.to_numpy())

                #del X_val
                #gc.collect()
            except MemoryError:
                logger.error(f'Memory error occurred while training {clf_name} with params {params}')
                score = 0
                failed = True
            except Exception as e:
                logger.error(f'Error occurred while training {clf_name} with params {params}')
                logger.error(e)
                score = 0
                failed = True
            finally:
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(f'(NEW BEST) Validation score for {clf_name} with params {params}: {score}')
                else:
                    logger.info(f'Validation score for {clf_name} with params {params}: {score}')

            # Update the checkpoint state
            grid_search_state[clf_name]['best_score'] = best_score
            grid_search_state[clf_name]['best_params'] = best_params
            grid_search_state[clf_name]['tried_params'].append(params)
            save_checkpoint(grid_search_state, checkpoint_file)


        if clf_name == 'NN':
            best_classifiers[clf_name] = NeuralNetwork(input_dim=X_shape, **best_params)
            best_classifiers[clf_name].compile(optim.Adam(best_classifiers[clf_name].parameters(), lr=0.001))
        elif clf_name == 'MLP':
            best_classifiers[clf_name] = PyTorchMLPClassifier(input_size=X_shape, num_classes=8, **best_params)
        else:
            best_classifiers[clf_name] = clf_class(**best_params)

        try:
            # If classifier is LinearSVC, we need to convert data to numpy
            if clf_name in ['NN', 'MLP','LinearSVC']:
                best_classifiers[clf_name].fit(X_train.compute().to_numpy(), y_train.compute())
            elif clf_name in ['RandomForest']:
                best_classifiers[clf_name].fit(X_train.compute().to_numpy(), y_train.compute().to_numpy())
            else:
                best_classifiers[clf_name].fit(X_train.compute(), y_train.compute())
        except MemoryError:
            logger.error(f'Memory error occurred while training {clf_name} with best parameters')
        except Exception as e:
            logger.error(f'Error occurred while training {clf_name} with best parameters')
            logger.error(e)
        #X_train = np.load(f'{args.experiment_dir}/{args.feature}/X_train.npy')
        #del X_train
        #gc.collect()
        logger.info(f'Best parameters for {clf_name}: {best_params}')

        try:
            #X_val = np.load(f'{args.experiment_dir}/{args.feature}/X_val.npy')
            if clf_name in ['NN', 'MLP']:
                y_pred = np.argmax(best_classifiers[clf_name].predict_proba(X_val.compute().to_numpy()), axis=1)
                logger.info(
                    f'Validation score for {clf_name}: {balanced_accuracy_score(y_val.compute().to_numpy(), y_pred)}')
            elif clf_name in ['RandomForest']:
                y_pred = best_classifiers[clf_name].predict(X_val.compute().to_numpy())
                logger.info(
                    f'Validation score for {clf_name}: {balanced_accuracy_score(y_val.compute().to_numpy(), y_pred)}')
            else:
                y_pred = best_classifiers[clf_name].predict(X_val.compute())
                logger.info(
                    f'Validation score for {clf_name}: {balanced_accuracy_score(y_val.compute().to_numpy(), y_pred.to_numpy())}')
        except MemoryError:
            logger.error(f'Memory error occurred while evaluating {clf_name} on validation set')
        except Exception as e:
            logger.error(f'Error occurred while evaluating {clf_name} on validation set')
            logger.error(e)


    for clf_name, best_clf in best_classifiers.items():
        if clf_name in ['NN', 'MLP','LinearSVC']:
            best_clf.fit(X_train.compute().to_numpy(), y_train.compute())
        elif clf_name in ['RandomForest']:
            best_clf.fit(X_train.compute().to_numpy(), y_train.compute().to_numpy())
        else:
            best_clf.fit(X_train.compute(), y_train.compute())

        if clf_name in ['NN', 'MLP']:
            y_pred = np.argmax(best_clf.predict_proba(X_test.compute().to_numpy()), axis=1)
            logger.info(f'Test score for {clf_name}: {balanced_accuracy_score(y_test.compute().to_numpy(), y_pred)}')
        elif clf_name in ['RandomForest']:
            y_pred = best_clf.predict(X_test.compute().to_numpy())
            logger.info(f'Test score for {clf_name}: {balanced_accuracy_score(y_test.compute().to_numpy(), y_pred)}')
        else:
            y_pred = best_clf.predict(X_test.compute())
            logger.info(f'Test score for {clf_name}: {balanced_accuracy_score(y_test.compute().to_numpy(), y_pred.to_numpy())}')










