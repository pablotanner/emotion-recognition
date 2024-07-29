import argparse
import itertools
import logging
import joblib
from cuml.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
import shap
from sklearn.metrics import confusion_matrix
# Look how the relative gain in score fusion (stacking) changes with the number of classifiers/feature types

"""
Contains code for Relative Score Fusion Gain Experiment in Section 6.3.3 and also for the score fusion importances analysis
in Section 7.2
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training one classifier with all feature types')
    parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                        default='/local/scratch/ptanner/relative_gain_experiments')
    #parser.add_argument('--use-concat', action='store_true', help='Use concatenated probabilities', default=False)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])
    logger.info("Starting Experiment")

    # Load the probabilities from the hybrid fusion experiment
    probabilities_val = np.load('/local/scratch/ptanner/hybrid_fusion_experiments/probabilities_val.npy', allow_pickle=True).item()
    probabilities_test = np.load('/local/scratch/ptanner/hybrid_fusion_experiments/probabilities_test.npy', allow_pickle=True).item()

    conc_path = '/local/scratch/ptanner/concatenated_experiment'

    probabilities_val['concat'] = np.load(f'{conc_path}/probabilities_val.npy', allow_pickle=True).item()['MLP']
    probabilities_test['concat'] = np.load(f'{conc_path}/probabilities_test.npy', allow_pickle=True).item()['MLP']

    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=1, class_weight='balanced'))
    ])

    # Start with hog probabilities, then add pdm, then add landmarks_3d, then add embedded and finally facs (Basic Pool)
    models = ['hog', 'pdm', 'landmarks_3d', 'embeddings', 'facs']
    # If Mixed feature type should be used (Extended Pool)
    #models = ['hog', 'pdm', 'landmarks_3d', 'embeddings', 'facs','concat']
    # If performing the importance experiment (only use the classifiers belonging to opimal subset):
    #models = ['hog', 'pdm', 'embeddings', 'facs']


    def evaluate_subsets():
        """
        Evaluates all possible subsets of classifier pool (for relative gain experiments)
        Best subset can be found by looking at the highest accuracy
        """
        increased_accuracy = []

        # Generate all non-empty subsets of models
        for subset_size in range(1, len(models) + 1):
            for subset in itertools.combinations(models, subset_size):
                logger.info(f"Using models: {subset}")
                X_stack = np.concatenate([probabilities_val[model] for model in subset], axis=1)
                stacking_pipeline.fit(X_stack, y_val)

                balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
                logger.info(
                    f"Balanced Accuracy of stacking classifier with models {subset} (Validation Set): {balanced_accuracy}")

                X_stack_test = np.concatenate([probabilities_test[model] for model in subset], axis=1)
                test_accuracy = stacking_pipeline.score(X_stack_test, y_test)
                logger.info(f"Accuracy of stacking classifier with models {subset} (Test Set): {test_accuracy}")

                increased_accuracy.append((subset, test_accuracy))

        return increased_accuracy

    # This function is called to get the optimal subset of basic/extended classifier pool in Section 6.3.3
    #evaluate_subsets()

    # This code was used to extract the SHAP importances for the "Score Fusion Importances" in Section 7.2
    # Note: Requires the optimal set to have been found first (models) and also the stacking pipeline to be trained
    def score_fusion_importances(y_val, stacking_pipeline, models):
        """
        Extracts SHAP values for the best stacking classifier

        """
        X_stack = np.concatenate([probabilities_val[model] for model in models], axis=1)
        stacking_pipeline.fit(X_stack, y_val)
        explainer = shap.Explainer(stacking_pipeline.named_steps['log_reg'], X_stack)

        # Only look at the correctly classified samples
        correct_indices = np.where(stacking_pipeline.predict(X_stack) == y_val)[0]

        X_stack = X_stack[correct_indices]

        shap_values = explainer(X_stack)

        joblib.dump(shap_values, f'SV_correct.joblib')



    logger.info("Experiment Finished")