import argparse
import gc
import logging
import os
import numpy as np
#from cuml.decomposition import IncrementalPCA as PCA
#from cuml.preprocessing import StandardScaler, MinMaxScaler
from cuml.svm import LinearSVC
from sklearn.decomposition import IncrementalPCA as PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from cuml.linear_model import LogisticRegression as CumlLogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import compute_class_weight
from src.model_training import SVC
from src.data_processing.rolf_loader import RolfLoader
from src.model_training.torch_neural_network import NeuralNetwork
import joblib
import cupy as cp
from src.model_training.torch_mlp import PyTorchMLPClassifier as MLP

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
parser.add_argument('--no-normalization', action='store_true', help="Don't use any normalization/standardization")
args = parser.parse_args()

feature_types = {
    'landmarks_3d': 'linear',
    #'facs_intensity': 'linear',
    #'facs_presence': 'linear',
    'hog': 'nonlinear',
    #'facenet': 'nonlinear',
    #'sface': 'nonlinear',
    'facs': 'nonlinear',
    'embedded': 'nonlinear',
    'nonrigid_face_shape': 'nonlinear'
}

def load_and_concatenate_features(dataset_type):
    logger.info('Loading Data')

    path = f'{args.experiment_dir}/{dataset_type}_concatenated_features.npy'

    names_path = f'{args.experiment_dir}/feature_names.npy'

    if os.path.exists(path) and os.path.exists(names_path):
        return path, np.load(names_path)

    X_list = []
    feature_names = []

    for feature in feature_types.keys():
        logger.info(f'Loading {feature}...')

        # Use memory mapping to load data
        file_path = f'{args.experiment_dir}/{dataset_type}_{feature}.npy'

        if args.load_gpu:
            data = cp.load(file_path, mmap_mode='r').astype(np.float32)
        else:
            data = np.load(file_path, mmap_mode='r').astype(np.float32)
        X_list.append(data)
        f_names = [f'{feature}_{i}' for i in range(data.shape[1])]
        feature_names.extend(f_names)

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
    np.save(names_path, feature_names)

    logger.info('Data saved')

    return path, feature_names


def filter_selection(X_train_path, X_val_path, X_test_path, y_train, k_features=200):
    X_selected_train_path = f'{args.experiment_dir}/train_selected.npy'
    X_selected_val_path = f'{args.experiment_dir}/val_selected.npy'
    X_selected_test_path = f'{args.experiment_dir}/test_selected.npy'


    if os.path.exists(X_selected_train_path):
        logger.info('Selected features already exist')
        return X_selected_train_path, X_selected_val_path, X_selected_test_path
    logger.info('Selecting features...')
    selector = SelectKBest(f_classif, k=k_features)


    X_train = selector.fit_transform(np.load(X_train_path).astype(np.float32), y_train)
    X_val = selector.transform(np.load(X_val_path).astype(np.float32))
    X_test = selector.transform(np.load(X_test_path).astype(np.float32))


    np.save(X_selected_train_path, X_train)
    np.save(X_selected_val_path, X_val)
    np.save(X_selected_test_path, X_test)

    return X_selected_train_path, X_selected_val_path, X_selected_test_path


def linear_selection(X_train_path, X_val_path, X_test_path, feature_names, y_train):
    X_selected_train_path = f'{args.experiment_dir}/train_selectedFM.npy'
    X_selected_val_path = f'{args.experiment_dir}/val_selectedFM.npy'
    X_selected_test_path = f'{args.experiment_dir}/test_selectedFM.npy'

    if os.path.exists(X_selected_train_path):
        logger.info('Selected features FM already exist')
        return X_selected_train_path, X_selected_val_path, X_selected_test_path
    logger.info('Selecting features from Model...')

    from sklearn.feature_selection import SelectFromModel
    lsvc = LinearSVC(C=0.01, penalty='l1', class_weight='balanced')
    lsvc.fit(np.load(X_train_path).astype(np.float32), y_train)
    selector = SelectFromModel(lsvc, prefit=True)

    try:
        feature_names = [feature_names[i] for i in selector.get_support(indices=True)]
        np.save(f'{args.experiment_dir}/feature_names_SFM.npy', feature_names)
        logger.info('SFM Feature names saved')
    except:
        logger.info('SFM Feature names not saved')

    X_train = selector.transform(np.load(X_train_path).astype(np.float32))
    X_val = selector.transform(np.load(X_val_path).astype(np.float32))
    X_test = selector.transform(np.load(X_test_path).astype(np.float32))

    np.save(X_selected_train_path, X_train)
    np.save(X_selected_val_path, X_val)
    np.save(X_selected_test_path, X_test)

    return X_selected_train_path, X_selected_val_path, X_selected_test_path



def preprocess_and_save_features(X_train, X_val, X_test, feature_name):
    # Check if the feature is already preprocessed
    #if f'train_{feature_name}.npy' in os.listdir(args.experiment_dir):
       #logger.info(f'{feature_name} already preprocessed')
        #return

    X_train = np.array(X_train).astype(np.float32)
    X_val = np.array(X_val).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)


    if args.no_normalization:
        logger.info('Skipping normalization')
    else:
        # Initialize Scaler
        logger.info(f'Scaling {feature_name}...')

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)


    # Step 2: Dimensionality Reduction
    if X_train.shape[1] > 50:
        logger.info(f'Dimensionality Reduction for {feature_name}...')
        pca_components = {
            'landmarks_3d': 100,
            'hog': 100,
            'embedded': 100,
        }
        pca = PCA(n_components=pca_components[feature_name])

        X_train = pca.fit_transform(X_train)
        # Save the PCA model
        joblib.dump(pca, f'{args.experiment_dir}/pca_models/{feature_name}_pca.joblib')
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

            preprocess_and_save_features(
                feature_splits_dict['train'][feature_name],
                feature_splits_dict['val'][feature_name],
                feature_splits_dict['test'][feature_name],
                feature_name,
            )

    for feature in feature_types.keys():
        continue
        if feature in ['facs', 'embedded']:
            preprocess_and_save_features(
                np.load(f'train_{feature}_features.npy').astype(np.float32),
                np.load(f'val_{feature}_features.npy').astype(np.float32),
                np.load(f'test_{feature}_features.npy').astype(np.float32),
                feature,
            )
        else:
            preprocess_and_save_features(
                np.load(f'{args.experiment_dir}/unprocessed/train_{feature}.npy').astype(np.float32),
                np.load(f'{args.experiment_dir}/unprocessed/val_{feature}.npy').astype(np.float32),
                np.load(f'{args.experiment_dir}/unprocessed/test_{feature}.npy').astype(np.float32),
                feature,
            )

    gc.collect()
    logger.info(f'Preparing concatenated data')
    y_train = np.load(f'y_train.npy')

    X_train_path, feature_names = load_and_concatenate_features('train')
    X_val_path, _ = load_and_concatenate_features('val')
    X_test_path, _ = load_and_concatenate_features('test')

    feature_names = np.array(feature_names)
    X_train_path, X_val_path, X_test_path = linear_selection(X_train_path, X_val_path, X_test_path, feature_names, y_train)

    #X_train_path, X_val_path, X_test_path = filter_selection(X_train_path, X_val_path, X_test_path, y_train, k_features=200)

    logger.info(f'Loading concatenated data...')
    
    X_train = np.load(X_train_path)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    nn = NeuralNetwork(input_dim=X_train.shape[1],  class_weight=class_weights, num_epochs=20, batch_size=128)
    #linearSVC = LinearSVC(class_weight='balanced', C=0.1, probability=True)
    #rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight=class_weights)
    #lr = CumlLogisticRegression(class_weight='balanced', C=1)
    svm = SVC(class_weight='balanced', probability=True, kernel='rbf', C=1)
    #mlp = MLP(batch_size=128, num_epochs=30, hidden_size=256, input_size=X_train.shape[1], class_weight=class_weights, learning_rate=0.01, num_classes=8)
    nn.__class__.__name__ = 'NeuralNetwork'
    #rf.__class__.__name__ = 'RandomForestClassifier'
    svm.__class__.__name__ = 'SVC'
    #lr.__class__.__name__ = 'LogisticRegression'
    #linearSVC.__class__.__name__ = 'LinearSVC'
    #mlp.__class__.__name__ = 'MLP'


    models = [
        nn,
        #lr,
        #linearSVC,
        #rf,
        svm,
    ]

    probabilities_val = {}
    probabilities_test = {}



    y_val = np.load(f'y_val.npy')
    X_val = np.load(X_val_path)

    X_test = np.load(X_test_path)
    y_test = np.load(f'y_test.npy')

    for model in models:
        #if os.path.exists(f'{args.experiment_dir}/models/{model.__class__.__name__}.joblib'):
            #logger.info(f'Loading {model.__class__.__name__}...')
            #model = joblib.load(f'{args.experiment_dir}/models/{model.__class__.__name__}.joblib')
        #else:
        logger.info(f'Training {model.__class__.__name__}...')
        model.fit(X_train, y_train)
            #joblib.dump(model, f'{args.experiment_dir}/models/{model.__class__.__name__}.joblib')
        proba = model.predict_proba(X_val)
        bal_acc_val = balanced_accuracy_score(y_val, np.argmax(proba, axis=1))
        logger.info(f'Balanced Accuracy of {model.__class__.__name__} (Validation Set): {bal_acc_val}')
        probabilities_val[model.__class__.__name__] = proba

        proba_test = model.predict_proba(X_test)
        bal_acc_test = balanced_accuracy_score(y_test, np.argmax(proba_test, axis=1))
        logger.info(f'Balanced Accuracy of {model.__class__.__name__} (Test Set): {bal_acc_test}')
        probabilities_test[model.__class__.__name__] = proba_test
    
    del X_train, y_train
    del X_val

    gc.collect()

    def evaluate_stacking(probabilities, y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([('scaler', StandardScaler()),('log_reg', LogisticRegression(C=1,  class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        #stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        #logger.info(f"Accuracy of stacking classifier (Validation Set): {stacking_accuracy}")

        balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
        logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")



        return stacking_pipeline

    # Use stacking
    stacking_pipeline = evaluate_stacking(probabilities_val, y_val)

    # Evaluate test set with stacking pipeline
    X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)
    test_balanced_accuracy = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_test_stack))
    logger.info(f"Balanced Accuracy of stacking classifier (Test Set): {test_balanced_accuracy}")