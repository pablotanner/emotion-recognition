import argparse
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from cuml.svm import LinearSVC, SVC
from cuml.preprocessing import StandardScaler
#from cuml.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from src.model_training.torch_mlp import PyTorchMLPClassifier as MLP
from src.model_training.torch_neural_network import NeuralNetwork

def get_tuned_classifiers(feature, class_weights, input_dim):
    """
    Returns a dictionary of tuned classifiers for a given feature.

    Classifiers were tuned by running grid search for each classifier and each feature (with StandardScaler).
    """
    if feature == 'pdm':
        return {
            'SVC': SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'),
            'LinearSVC': LinearSVC(C=0.1, probability=True, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=20,
                                                   min_samples_split=2, criterion='entropy', class_weight='balanced'),
            'LogisticRegression': LogisticRegression(C=10, class_weight='balanced'),
            'MLP': MLP(hidden_size=256, batch_size=32, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                       num_classes=8, input_size=input_dim),
            'NN': NeuralNetwork(batch_size=128, num_epochs=30, class_weight=class_weights, input_dim=input_dim)
        }
    elif feature == 'facs':
        return {
            'SVC':  SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'),
            'LinearSVC': LinearSVC(C=0.1, probability=True, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=20,
                                                   min_samples_split=2, criterion='entropy', class_weight='balanced'),
            'LogisticRegression': LogisticRegression(C=10, class_weight='balanced'),
            'MLP': MLP(hidden_size=256, batch_size=32, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                       num_classes=8, input_size=input_dim),
            'NN': NeuralNetwork(batch_size=128, num_epochs=30, class_weight=class_weights, input_dim=input_dim)
        }
    elif feature == 'landmarks_3d':
        return {
            'SVC': SVC(C=10, probability=True, kernel='rbf', class_weight='balanced'),
            'LinearSVC': LinearSVC(C=0.1, probability=True, class_weight='balanced'),
            'RandomForest': None,
            'LogisticRegression': LogisticRegression(C=10, class_weight='balanced'),
            'MLP': None,
            'NN': None,
        }
    elif feature == 'embedded':
        return {
            'SVC': SVC(C=0.5, probability=True, kernel='rbf', class_weight='balanced'),
            'LinearSVC': LinearSVC(C=0.1, probability=True, class_weight='balanced'),
            'RandomForest': None,
            'LogisticRegression': LogisticRegression(C=1, class_weight='balanced'),
            'MLP': MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                       num_classes=8, input_size=input_dim),
            'NN': NeuralNetwork(batch_size=128, num_epochs=20, class_weight=class_weights, input_dim=input_dim)
        }
    elif feature == 'hog':
        return {
            'SVC': SVC(C=0.1, probability=True, kernel='rbf', class_weight='balanced'),
            'LinearSVC': LinearSVC(C=0.1, probability=True, class_weight='balanced'),
            'RandomForest': None,
            'LogisticRegression': LogisticRegression(C=1, class_weight='balanced'),
            'MLP': MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=20,
                       num_classes=8, input_size=input_dim),
            'NN': None
        }
    else:
        raise ValueError(f"Feature {feature} not supported.")

feature_paths = {
    'hog': {
        'train': 'pca_train_hog_features.npy',
        'val': 'pca_val_hog_features.npy',
        'test': 'pca_test_hog_features.npy'
    },
    'landmarks_3d': {
        'train': 'train_spatial_features.npy',
        'val': 'val_spatial_features.npy',
        'test': 'test_spatial_features.npy'
    },
    'pdm': {
        'train': 'train_pdm_features.npy',
        'val': 'val_pdm_features.npy',
        'test': 'test_pdm_features.npy'
    },
    'facs': {
        'train': 'train_facs_features.npy',
        'val': 'val_facs_features.npy',
        'test': 'test_facs_features.npy'
    },
    'embedded': {
        'train': 'train_embedded_features.npy',
        'val': 'val_embedded_features.npy',
        'test': 'test_embedded_features.npy'
    }
}

parser = argparse.ArgumentParser(
    description='Training same feature on different classifiers or different features on same classifier')
parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                    default='/local/scratch/ptanner/cf_experiments')
parser.add_argument('--feature', type=str, help='Feature to use for training', default='pdm')
args = parser.parse_args()

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])
    logger.info("Starting Experiment")

    feature = args.feature


    probabilities_val = {}
    probabilities_test = {}

    X_train_path = feature_paths[feature]['train']
    X_val_path = feature_paths[feature]['val']
    X_test_path = feature_paths[feature]['test']

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    input_shape = np.load(X_test_path).shape[1]
    num_classes = len(np.unique(y_train))

    pipelines = []


    for name, classifier in get_tuned_classifiers(feature, class_weights, input_shape).items():
        if classifier is not None:
            pipelines.append(Pipeline([
                ('scaler', StandardScaler()),
                (name, classifier)
            ]))

    for pipeline in pipelines:
        logger.info(f"Training with {pipeline.steps[-1][0]}")
        pipeline.fit(np.load(X_train_path), y_train)
        probabilities_val[pipeline] = pipeline.predict_proba(np.load(X_val_path))
        probabilities_test[pipeline] = pipeline.predict_proba(np.load(X_test_path))
        bal_acc_val = balanced_accuracy_score(y_val, np.argmax(probabilities_val[pipeline], axis=1))
        bal_acc_test = balanced_accuracy_score(y_test, np.argmax(probabilities_test[pipeline], axis=1))
        logger.info(f"Balanced Accuracy on Validation: {bal_acc_val}")
        logger.info(f"Balanced Accuracy on Test: {bal_acc_test}")

    # Stacking
    def evaluate_stacking(probabilities, y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([('log_reg', LogisticRegression(C=1, class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        logger.info(f"Accuracy of stacking classifier (Validation Set): {stacking_accuracy}")

        balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
        logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")

        # Return the stacking pipeline
        return stacking_pipeline

    stacking_pipeline = evaluate_stacking(probabilities_val, y_val)

    # Evaluate the stacking classifier on the test set
    X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)
    test_accuracy = stacking_pipeline.score(X_test_stack, y_test)
    logger.info(f"Accuracy of stacking classifier (Test Set): {test_accuracy}")

    logger.info("Experiment Finished")




