import argparse
import logging
import os
import joblib
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from src.model_training import SVC
from src.util.data_paths import get_data_path
from imblearn.under_sampling import RandomUnderSampler
from cuml.ensemble import RandomForestClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Importance Experiment')
    parser.add_argument('--feature', type=str, help='Feature Type',
                        default='facs')

    experiment_dir = '/local/scratch/ptanner/feature_importance_experiments'
    pca_dir = '/local/scratch/ptanner/concatenated_experiment/pca_models'

    args = parser.parse_args()

    feature = args.feature

    # Check if exp directory and feature directory exists
    if not os.path.exists(f'{experiment_dir}/{args.feature}'):
        os.makedirs(f'{experiment_dir}/{args.feature}')


    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{experiment_dir}/{feature}/experiment.log'),
                            logging.StreamHandler()
                        ])


    # Check if undersampled data already exists
    if os.path.exists(f'{experiment_dir}/{feature}/X_train.npy'):
        logger.info(f"Undersampled data for {feature} already exists. Loading")
        X_train = np.load(f'{experiment_dir}/{feature}/X_train.npy')
        y_train = np.load(f'{experiment_dir}/{feature}/y_train.npy')
        y_test = np.load(f'{experiment_dir}/{feature}/y_test.npy')
    else:
        logger.info(f"Undersampled data for {feature} does not exist. Generating")
        # Concatenate train, val, test for undersampling
        X = np.concatenate([np.load(get_data_path('train', args.feature)), np.load(get_data_path('val', args.feature)), np.load(get_data_path('test', args.feature))]).astype(np.float32)

        y = np.concatenate([np.load('y_train.npy'), np.load('y_val.npy'), np.load('y_test.npy')])
        # Undersample
        rus = RandomUnderSampler(random_state=42)
        X_train, y_train = rus.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        logger.info(f"X_train shape: {X_train.shape}. Saving")
        np.save(f'{experiment_dir}/{feature}/X_train.npy', X_train)
        np.save(f'{experiment_dir}/{feature}/y_train.npy', y_train)
        np.save(f'{experiment_dir}/{feature}/X_test.npy', X_test)
        np.save(f'{experiment_dir}/{feature}/y_test.npy', y_test)


    # Coincidentally, all 3 (4 with concat) feature types have same SVC RBF params
    if os.path.exists(f'{experiment_dir}/{feature}/classifier.joblib'):
        logger.info(f"Classifier for {args.feature} already exists. Loading")
        svc = joblib.load(f'{experiment_dir}/{feature}/classifier.joblib')
    else:
        logger.info(f"Training classifier on {args.feature} features")
        svc = SVC(C=1, kernel='rbf', class_weight='balanced')
        # Fit SVC
        svc.fit(X_train, y_train)
        joblib.dump(svc, f'{experiment_dir}/{feature}/classifier.joblib')


    if os.path.exists(f'{experiment_dir}/{feature}/explainer.joblib'):
        logger.info(f"Explainer for {args.feature} already exists. Loading")
        explainer = joblib.load(f'{experiment_dir}/{feature}/explainer.joblib')
    else:
        logger.info(f"Generating Explainer")
        # Generate Explainer
        #explainer = KernelExplainer(model=svc.predict, data=X_train, random_state=42, is_gpu_model=True, dtype=np.float32)
        #explainer = PermutationExplainer(model=svc.predict, data=X_train, random_state=42, is_gpu_model=True, dtype=np.float32)
        explainer = shap.Explainer(svc.predict, X_train)
        #explainer = KernelExplainer(model=svc.predict,data=X_train,is_gpu_model=True,random_state=42,dtype=np.float32)
        joblib.dump(explainer, f'{experiment_dir}/{feature}/explainer.joblib')

    # Delete Unnecessary Variables
    del X_train, y_train, svc

    logger.info(f"Generating SHAP Values")

    # Get SHAP Values
    shap_values = explainer(X_test)

    if args.feature == 'landmarks_3d':
        pca = joblib.load(f'{pca_dir}/landmarks_3d_pca.joblib')
        shap_values = pca.inverse_transform(shap_values)
    elif args.feature == 'concatenated':
        emb_pca = joblib.load(f'{pca_dir}/embedded_pca.joblib')
        lnd_pca = joblib.load(f'{pca_dir}/concatenated_pca.joblib')
        hog_pca = joblib.load(f'{pca_dir}/hog_pca.joblib')
        # Not implemented for now

    # Dump SHAP Values
    joblib.dump(shap_values, f'{experiment_dir}/{feature}/shap_values.joblib')







