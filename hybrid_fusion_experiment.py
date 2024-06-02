import numpy as np
import argparse
from cuml.preprocessing import StandardScaler as CUMLStandardScaler
from cuml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight

from classifier_vs_feature_experiment import feature_paths
from src.model_training.torch_neural_network import NeuralNetwork


# Experiment where I use hybrid of different features and classifiers to basically optimize score fusion results


if __name__ == '__main__':
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    def prepare_facs():
        pipeline = Pipeline([
            ('scaler', CUMLStandardScaler()),
            ('random_forest',
             RandomForestClassifier(n_estimators=400, max_depth=None, max_features=None, min_samples_split=10,
                                    criterion='gini', class_weight='balanced'))
        ])

        pipeline.fit(np.load(feature_paths['facs']['train']).astype(np.float32), y_train)

        return pipeline


    def prepare_lnd():
        input_dim = np.load(feature_paths['landmarks_3d']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
        ])

        pipeline.fit(np.load(feature_paths['landmarks_3d']['train']).astype(np.float32), y_train)

        return pipeline

    def prepare_pdm():
        input_dim = np.load(feature_paths['pdm']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', CUMLStandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
        ])

        pipeline.fit(np.load(feature_paths['pdm']['train']).astype(np.float32), y_train)

        return pipeline

    def prepare_emb():
        input_dim = np.load(feature_paths['embedded']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', CUMLStandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=30, class_weight=class_weights, input_dim=input_dim))
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


    parser = argparse.ArgumentParser(description='Hybrid Fusion Experiment')
    parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                        default='/local/scratch/ptanner/hybrid_fusion_experiments')
    args = parser.parse_args()

    probabilities_val = {}
    probabilities_test = {}

    facs_pipeline = prepare_facs()
    lnd_pipeline = prepare_lnd()
    pdm_pipeline = prepare_pdm()
    emb_pipeline = prepare_emb()
    hog_pipeline = prepare_hog()

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

    # Stacking
    X_stack_val = np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1)

    stacking_pipeline = Pipeline([

        ('log_reg', LogisticRegression(C=1, class_weight='balanced'))
    ])

    stacking_pipeline.fit(X_stack_val, y_val)

    balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack_val))

    print(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")

    X_stack_test = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

    accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_stack_test))

    print(f"Balanced Accuracy of stacking classifier (Test Set): {balanced_accuracy}")

    print("Finished Hybrid Fusion Experiment")








