import logging
import numpy as np
import argparse
from cuml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from cuml.linear_model import LogisticRegression as CULogisticRegression

from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from src.model_training import SVC
from sklearn.utils import compute_class_weight
from classifier_vs_feature_experiment import feature_paths
from src.model_training.torch_neural_network import NeuralNetwork
from src.model_training.torch_mlp import PyTorchMLPClassifier as MLP


# Experiment where I use hybrid of different features and classifiers to basically optimize score fusion results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid Fusion Experiment')
    parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                        default='/local/scratch/ptanner/hybrid_fusion_experiments')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    def prepare_facs():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', CULogisticRegression(C=10, class_weight='balanced'))
        ])

        pipeline.fit(np.load(feature_paths['facs']['train']).astype(np.float32), y_train)

        return pipeline


    def prepare_lnd():
        #input_dim = np.load(feature_paths['landmarks_3d']['test']).shape[1]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
            ('svc', SVC(C=10, probability=True, kernel='rbf', class_weight='balanced'))
        ])

        pipeline.fit(np.load(feature_paths['landmarks_3d']['train']).astype(np.float32), y_train)

        return pipeline

    def prepare_pdm():
        #input_dim = np.load(feature_paths['pdm']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
            ('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
        ])

        pipeline.fit(np.load(feature_paths['pdm']['train']).astype(np.float32), y_train)

        return pipeline

    def prepare_emb():
        input_dim = np.load(feature_paths['embedded']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
            ('mlp', MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                       num_classes=8, input_size=input_dim))
        ])

        pipeline.fit(np.load(feature_paths['embedded']['train']).astype(np.float32), y_train)

        return pipeline

    def prepare_hog():
        input_dim = np.load(feature_paths['hog']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=30, class_weight=class_weights, input_dim=input_dim))
        ])

        pipeline.fit(np.load(feature_paths['hog']['train']).astype(np.float32), y_train)

        return pipeline



    probabilities_val = {}
    probabilities_test = {}

    logger.info("Starting Hybrid Fusion Experiment")
    logger.info("Fitting FACS")
    facs_pipeline = prepare_facs()
    logger.info("Fitting Landmarks 3D")
    lnd_pipeline = prepare_lnd()
    logger.info("Fitting PDM")
    pdm_pipeline = prepare_pdm()
    logger.info("Fitting Embedded")
    emb_pipeline = prepare_emb()
    logger.info("Fitting HOG")
    hog_pipeline = prepare_hog()

    # Get individual accuracies
    balanced_accuracy = balanced_accuracy_score(y_val, facs_pipeline.predict(np.load(feature_paths['facs']['val']).astype(np.float32)))
    logger.info(f"Balanced Accuracy of FACS classifier (Validation Set): {balanced_accuracy}")

    balanced_accuracy = balanced_accuracy_score(y_val, lnd_pipeline.predict(np.load(feature_paths['landmarks_3d']['val']).astype(np.float32)))
    logger.info(f"Balanced Accuracy of Landmarks 3D classifier (Validation Set): {balanced_accuracy}")

    balanced_accuracy = balanced_accuracy_score(y_val, pdm_pipeline.predict(np.load(feature_paths['pdm']['val']).astype(np.float32)))
    logger.info(f"Balanced Accuracy of PDM classifier (Validation Set): {balanced_accuracy}")

    balanced_accuracy = balanced_accuracy_score(y_val, emb_pipeline.predict(np.load(feature_paths['embedded']['val']).astype(np.float32)))
    logger.info(f"Balanced Accuracy of Embedded classifier (Validation Set): {balanced_accuracy}")

    balanced_accuracy = balanced_accuracy_score(y_val, hog_pipeline.predict(np.load(feature_paths['hog']['val']).astype(np.float32)))
    logger.info(f"Balanced Accuracy of HOG classifier (Validation Set): {balanced_accuracy}")



    probabilities_val['facs'] = facs_pipeline.predict_proba(np.load(feature_paths['facs']['val']).astype(np.float32))
    probabilities_val['landmarks_3d'] = lnd_pipeline.predict_proba(np.load(feature_paths['landmarks_3d']['val']).astype(np.float32))
    probabilities_val['pdm'] = pdm_pipeline.predict_proba(np.load(feature_paths['pdm']['val']).astype(np.float32))
    probabilities_val['embedded'] = emb_pipeline.predict_proba(np.load(feature_paths['embedded']['val']).astype(np.float32))
    probabilities_val['hog'] = hog_pipeline.predict_proba(np.load(feature_paths['hog']['val']).astype(np.float32))

    probabilities_test['facs'] = facs_pipeline.predict_proba(np.load(feature_paths['facs']['test']).astype(np.float32))
    probabilities_test['landmarks_3d'] = lnd_pipeline.predict_proba(np.load(feature_paths['landmarks_3d']['test']).astype(np.float32))
    probabilities_test['pdm'] = pdm_pipeline.predict_proba(np.load(feature_paths['pdm']['test']).astype(np.float32))
    probabilities_test['embedded'] = emb_pipeline.predict_proba(np.load(feature_paths['embedded']['test']).astype(np.float32))
    probabilities_test['hog'] = hog_pipeline.predict_proba(np.load(feature_paths['hog']['test']).astype(np.float32))

    np.save(f'{args.experiment_dir}/probabilities_val.npy', probabilities_val)
    np.save(f'{args.experiment_dir}/probabilities_test.npy', probabilities_test)

    logger.info("Probabilities saved")

    # Stacking
    X_stack_val = np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1)

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=1, class_weight='balanced'))
    ])

    stacking_pipeline.fit(X_stack_val, y_val)

    balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack_val))

    logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")

    X_stack_test = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

    balanced_accuracy = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_stack_test))

    logger.info(f"Balanced Accuracy of stacking classifier (Test Set): {balanced_accuracy}")

    logger.info("Experiment Finished")








