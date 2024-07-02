import argparse
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from src.model_training import SVC
from src.model_training.torch_neural_network import NeuralNetwork
from src.model_training.torch_mlp import PyTorchMLPClassifier as MLP


# Experiment to get the raw confusion matrices for the individual features again

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw Feature Confusion Matrix Experiment')
    parser.add_argument('--feature', type=str, help='Feature Type',
                        default='pdm')
    experiment_dir = '/local/scratch/ptanner/raw_feature_cm_experiments'
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    logger.info(f"Training Classifier on {args.feature} features")
    if args.feature == 'landmarks_3d':
        X_train = np.load('train_spatial_features.npy').astype(np.float32)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(np.load('test_spatial_features.npy').astype(np.float32))

    elif args.feature == 'embeddings':
        X_train = np.load('train_embeddings.npy').astype(np.float32)
        input_dim = X_train.shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                       num_classes=8, input_size=input_dim))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(np.load('test_embeddings.npy').astype(np.float32))

    elif args.feature == 'hog':
        X_train = np.load('pca_train_hog_features.npy').astype(np.float32)
        input_dim = X_train.shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=20, class_weight=class_weights, input_dim=input_dim))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(np.load('pca_test_hog_features.npy').astype(np.float32))

    elif args.feature == 'concatenated':
        X_train = np.load('/local/scratch/ptanner/concatenated_experiment/train_concatenated_features.npy').astype(np.float32)
        input_dim = X_train.shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLP(hidden_size=256, batch_size=128, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                        num_classes=8, input_size=input_dim))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(np.load('/local/scratch/ptanner/concatenated_experiment/test_concatenated_features.npy').astype(np.float32))
    else:
        raise ValueError("Invalid Feature Type")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix for {args.feature} features")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    logger.info(f"Normalized Confusion Matrix for {args.feature} features")
    logger.info(np.around(cm_normalized, 2))

    np.save(f'{experiment_dir}/{args.feature}_cm.npy', cm)
    np.save(f'{experiment_dir}/{args.feature}_cm_normalized.npy', cm_normalized)



