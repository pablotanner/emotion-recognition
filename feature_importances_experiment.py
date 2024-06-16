import argparse
import logging
import os

import joblib
import numpy as np
#import shap
from cuml.explainer import PermutationExplainer, KernelExplainer
from sklearn.model_selection import train_test_split
from src.model_training import SVC
from cuml.svm import LinearSVC
from src.util.data_paths import get_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Importance Experiment')
    parser.add_argument('--feature', type=str, help='Feature Type',
                        default='facs')
    parser.add_argument('--clf', type=str, help='Classifier')
    parser.add_argument('--exp', type=str, help='Kernel Explainer or Permutation Explainer')


    experiment_dir = '/local/scratch/ptanner/feature_importance_experiments'
    pca_dir = '/local/scratch/ptanner/concatenated_experiment/pca_models'

    args = parser.parse_args()

    feature = args.feature
    clf = args.clf
    exp = args.exp


    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])

    # Might be PCA Reduced
    X_train = np.load(get_data_path('train', args.feature)).astype(np.float32)
    X_test = np.load(get_data_path('test', args.feature)).astype(np.float32)
    y_train = np.load('y_train.npy')

    #X = np.load(get_data_path('test', args.feature)).astype(np.float32)
    #y = np.load('y_test.npy')

    # Split into Train and Test

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Coincidentally, all 3 (4 with concat) feature types have same SVC RBF params
    if os.path.exists(f'{experiment_dir}/{args.feature}_{clf}.joblib'):
        logger.info(f"Classifier {clf} for {args.feature} already exists. Loading")
        svc = joblib.load(f'{experiment_dir}/{args.feature}_{clf}.joblib')
    else:
        logger.info(f"Training {clf} on {args.feature} features")
        if clf == 'svc':
            svc = SVC(C=1, kernel='rbf', class_weight='balanced')
        else:
            svc = LinearSVC(C=1, class_weight='balanced')
        # Fit SVC
        svc.fit(X_train, y_train)
        joblib.dump(svc, f'{experiment_dir}/{args.feature}_{clf}.joblib')


    if os.path.exists(f'{experiment_dir}/{args.feature}_{exp}.joblib'):
        logger.info(f"Explainer {exp} for {args.feature} already exists. Loading")
        explainer = joblib.load(f'{experiment_dir}/{args.feature}_{exp}.joblib')
    else:
        logger.info(f"Generating Explainer")
        # Generate Explainer
        if exp == 'kernel':
            explainer = KernelExplainer(model=svc.predict, data=X_train, random_state=42, is_gpu_model=True, dtype=np.float32)
        else:
            explainer = PermutationExplainer(model=svc.predict, data=X_train, random_state=42, is_gpu_model=True, dtype=np.float32)
        #explainer = shap.Explainer(svc.predict, X_train)
        #explainer = KernelExplainer(model=svc.predict,data=X_train,is_gpu_model=True,random_state=42,dtype=np.float32)
        joblib.dump(explainer, f'{experiment_dir}/{args.feature}_{exp}.joblib')

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
    joblib.dump(shap_values, f'{experiment_dir}/{args.feature}_shap_values.joblib')







