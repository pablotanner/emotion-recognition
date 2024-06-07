import argparse
import itertools
import logging
import joblib
from cuml.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
import shap
# Look how the relative gain in score fusion (stacking) changes with the number of classifiers/feature types

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


    probabilities_val['concat'] = np.load('/local/scratch/ptanner/concatenated_experiment/probabilities_val.npy', allow_pickle=True).item()['SVC']
    probabilities_test['concat'] = np.load('/local/scratch/ptanner/concatenated_experiment/probabilities_val.npy', allow_pickle=True).item()['SVC']

    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')


    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=1, class_weight='balanced'))
    ])

    # Start with hog probabilities, then add pdm, then add landmarks_3d, then add embedded and finally facs
    models = ['hog', 'pdm', 'landmarks_3d', 'embedded', 'facs', 'concat']



    def do_experiment():
        increased_accuracy = []
        for i in range(1, len(models) + 1):
            # Print the models being used
            logger.info(f"Using models: {models[:i]}")
            X_stack = np.concatenate([probabilities_val[model] for model in models[:i]], axis=1)
            stacking_pipeline.fit(X_stack, y_val)
            # stacking_accuracy = stacking_pipeline.score(X_stack, y_val)
            # logger.info(f"Accuracy of stacking classifier with {i} models (Validation Set): {stacking_accuracy}")

            balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
            logger.info(
                f"Balanced Accuracy of stacking classifier with {i} models (Validation Set): {balanced_accuracy}")

            X_stack_test = np.concatenate([probabilities_test[model] for model in models[:i]], axis=1)
            test_accuracy = stacking_pipeline.score(X_stack_test, y_test)
            logger.info(f"Accuracy of stacking classifier with {i} models (Test Set): {test_accuracy}")

            increased_accuracy.append(test_accuracy)
            # Save
            joblib.dump(stacking_pipeline, f'{args.experiment_dir}/stacking_pipeline_{i}.joblib')

        logger.info(f"Accuracy increase: {increased_accuracy}")


    def do_experiment_subset():
        increased_accuracy = []

        # Generate all non-empty subsets of models
        for subset_size in range(1, len(models) + 1):
            for subset in itertools.combinations(models, subset_size):
                # Generate all permutations of the current subset
                for permutation in itertools.permutations(subset):
                    logger.info(f"Using models: {permutation}")
                    X_stack = np.concatenate([probabilities_val[model] for model in permutation], axis=1)
                    stacking_pipeline.fit(X_stack, y_val)

                    #balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
                    #logger.info(
                       # f"Balanced Accuracy of stacking classifier with models {permutation} (Validation Set): {balanced_accuracy}")

                    X_stack_test = np.concatenate([probabilities_test[model] for model in permutation], axis=1)
                    test_accuracy = stacking_pipeline.score(X_stack_test, y_test)
                    logger.info(
                        f"Accuracy of stacking classifier with models {permutation} (Test Set): {test_accuracy}")

                    increased_accuracy.append((permutation, test_accuracy))

                    # Save the model
                    # joblib.dump(stacking_pipeline,
                               # f'{args.experiment_dir}/stacking_pipeline_{"_".join(permutation)}.joblib')

        # Optionally, you could return or save the increased_accuracy list to analyze later
        return increased_accuracy

    #do_experiment()

    increased_accuracy = do_experiment_subset()

    #X_stack = np.concatenate([probabilities_val[model] for model in models], axis=1)
    #X_stack_test = np.concatenate([probabilities_test[model] for model in models], axis=1)
    #stacking_pipeline.fit(X_stack, y_val)
    #explainer = shap.Explainer(stacking_pipeline.named_steps['log_reg'], X_stack)
    #shap_values_test = explainer(X_stack_test)

    #shap.bar_plot(shap_values_test, max_display=10)


    logger.info("Experiment Finished")