import argparse
import logging
import os

import joblib
import numpy as np
#import shap
from cuml.explainer import PermutationExplainer, KernelExplainer
from src.model_training import SVC
from src.util.data_paths import get_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Importance Experiment')
    parser.add_argument('--feature', type=str, help='Feature Type',
                        default='facs')
    experiment_dir = '/local/scratch/ptanner/feature_importance_experiments'
    pca_dir = '/local/scratch/ptanner/concatenated_experiment/pca_models'

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])

    # Might be PCA Reduced
    X_train = np.load(get_data_path('train', args.feature)).astype(np.float32)
    y_train = np.load('y_train.npy')

    X_test = np.load(get_data_path('test', args.feature)).astype(np.float32)

    # Coincidentally, all 3 (4 with concat) feature types have same SVC RBF params
    if os.path.exists(f'{experiment_dir}/{args.feature}_classifier.joblib'):
        logger.info(f"Classifier for {args.feature} already exists. Loading")
        svc = joblib.load(f'{experiment_dir}/{args.feature}_classifier.joblib')
    else:
        logger.info(f"Training SVC on {args.feature} features")
        svc = SVC(C=1, probability=True, kernel='rbf', class_weight='balanced')
        # Fit SVC
        svc.fit(X_train, y_train)
        joblib.dump(svc, f'{experiment_dir}/{args.feature}_classifier.joblib')



    logger.info(f"Generating Explainer")
    # Generate Explainer
    #explainer = PermutationExplainer(model=svc.predict, data=X_train, random_state=42)
    #explainer = shap.Explainer(svc.predict, X_train)
    explainer = KernelExplainer(svc.predict, X_train)

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







