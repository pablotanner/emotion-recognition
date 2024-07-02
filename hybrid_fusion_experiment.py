import logging
import numpy as np
import argparse
from cuml.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from src.model_training import SVC
from sklearn.utils import compute_class_weight
from src.model_training.torch_neural_network import NeuralNetwork
from src.model_training.torch_mlp import PyTorchMLPClassifier as MLP
from src.util.data_paths import get_data_path

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
        input_dim = np.load(get_data_path('test', 'facs')).shape[1]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('log_reg', LogisticRegression(C=0.1, class_weight='balanced'))
            #('rf', RandomForestClassifier(n_estimators=400, max_depth=20, class_weight='balanced'))
             # min_samples_split=2, criterion='gini', class_weight='balanced'))
            ('mlp', MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                        num_classes=8, input_size=input_dim))
        ])

        pipeline.fit(np.load(get_data_path('train', 'facs')).astype(np.float32), y_train)

        return pipeline


    def prepare_lnd():
        #input_dim = np.load(feature_paths['landmarks_3d']['test']).shape[1]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
            ('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
        ])

        pipeline.fit(np.load(get_data_path('train', 'landmarks_3d')).astype(np.float32), y_train)

        return pipeline

    def prepare_pdm():
        #input_dim = np.load(feature_paths['pdm']['test']).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('nn', NeuralNetwork(batch_size=128, num_epochs=50, class_weight=class_weights, input_dim=input_dim))
            ('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
        ])

        pipeline.fit(np.load(get_data_path('train', 'nonrigid_face_shape')).astype(np.float32), y_train)

        return pipeline

    def prepare_emb():
        input_dim = np.load(get_data_path('test', 'embeddings')).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            #('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
            #('svc', SVC(C=1, probability=True, kernel='rbf', class_weight='balanced'))
            ('mlp', MLP(hidden_size=256, batch_size=32, class_weight=class_weights, learning_rate=0.01, num_epochs=30,
                        num_classes=8, input_size=input_dim))
        ])

        pipeline.fit(np.load(get_data_path('train', 'embeddings')).astype(np.float32), y_train)

        return pipeline

    def prepare_hog():
        input_dim = np.load(get_data_path('test', 'hog')).shape[1]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('nn', NeuralNetwork(batch_size=128, num_epochs=20, class_weight=class_weights, input_dim=input_dim))
        ])


        pipeline.fit(np.load(get_data_path('train', 'hog')).astype(np.float32), y_train)

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
    logger.info("Fitting embeddings")
    emb_pipeline = prepare_emb()
    logger.info("Fitting HOG")
    hog_pipeline = prepare_hog()

    probabilities_val['facs'] = facs_pipeline.predict_proba(np.load(get_data_path('val', 'facs')).astype(np.float32))
    probabilities_val['landmarks_3d'] = lnd_pipeline.predict_proba(np.load(get_data_path('val', 'landmarks_3d')).astype(np.float32))
    probabilities_val['pdm'] = pdm_pipeline.predict_proba(np.load(get_data_path('val', 'nonrigid_face_shape')).astype(np.float32))
    probabilities_val['embeddings'] = emb_pipeline.predict_proba(np.load(get_data_path('val', 'embeddings')).astype(np.float32))
    probabilities_val['hog'] = hog_pipeline.predict_proba(np.load(get_data_path('val', 'hog')).astype(np.float32))

    for model in probabilities_val:
        balanced_accuracy = balanced_accuracy_score(y_val, np.argmax(probabilities_val[model], axis=1))
        logger.info(f"Balanced Accuracy of {model} classifier (Validation Set): {balanced_accuracy}")

    X_train_path = get_data_path('train', 'concat')

    probabilities_test['facs'] = facs_pipeline.predict_proba(np.load(get_data_path('test', 'facs')).astype(np.float32))
    probabilities_test['landmarks_3d'] = lnd_pipeline.predict_proba(np.load(get_data_path('test', 'landmarks_3d')).astype(np.float32))
    probabilities_test['pdm'] = pdm_pipeline.predict_proba(np.load(get_data_path('test', 'nonrigid_face_shape')).astype(np.float32))
    probabilities_test['embeddings'] = emb_pipeline.predict_proba(np.load(get_data_path('test', 'embeddings')).astype(np.float32))
    probabilities_test['hog'] = hog_pipeline.predict_proba(np.load(get_data_path('test', 'hog')).astype(np.float32))

    np.save(f'{args.experiment_dir}/probabilities_val.npy', probabilities_val)
    np.save(f'{args.experiment_dir}/probabilities_test.npy', probabilities_test)

    logger.info("Probabilities saved")

    # Stacking
    X_stack_val = np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1)
    X_stack_test = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    best_solver = None
    best_balanced_accuracy = 0
    best_stacking_pipeline = None
    for solver in solvers:
        logger.info(f"Training Stacking Classifier with solver: {solver}")
        stacking_pipeline = Pipeline([
            # ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(C=1, class_weight='balanced', solver=solver, max_iter=1000))
        ])
        stacking_pipeline.fit(X_stack_val, y_val)
        #balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack_val))
        #logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")


        balanced_accuracy = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_stack_test))

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_solver = solver
            best_stacking_pipeline = stacking_pipeline

        logger.info(f"Balanced Accuracy of stacking classifier (Test Set): {balanced_accuracy}")
        # Confusion Matrix

    logger.info(f"Best Solver: {best_solver}, With Balanced Accuracy: {best_balanced_accuracy}")

    cm = confusion_matrix(y_test, best_stacking_pipeline.predict(X_stack_test))
    # Standardize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    logger.info(f"Normalized Confusion Matrix: {cm_norm}")
    np.save(f'{args.experiment_dir}/cm_stacking.npy', cm)
    np.save(f'{args.experiment_dir}/cm_stacking_norm.npy', cm_norm)

    logger.info("Experiment Finished")








