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
from sklearn.metrics import confusion_matrix
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

    conc_path = '/local/scratch/ptanner/concatenated_experiment'

    probabilities_val['concat'] = np.load(f'{conc_path}/probabilities_val.npy', allow_pickle=True).item()['MLP']
    probabilities_test['concat'] = np.load(f'{conc_path}/probabilities_test.npy', allow_pickle=True).item()['MLP']

    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    #cm_matrices = {}
    #not_normalized = {}
    # Evaluate individual models with confusion matrices
    #for model in probabilities_test.keys():
    #    y_pred = np.argmax(probabilities_test[model], axis=1)
    #    cm = confusion_matrix(y_test, y_pred)
    #    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #   cm_matrices[model] = cm_normalized
    #    not_normalized[model] = cm
    #    print(f"Confusion Matrix for {model}")
    #    print(cm)

    #    print(f"Normalized Confusion Matrix for {model}")
    #    # Format the output
    #    print(np.around(cm_normalized, 2))

    # np.save('cm_matrices.npy', cm_matrices)
    # np.save('cm_matrices_not_normalized.npy', not_normalized)
    #exit(0)

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(C=1, class_weight='balanced'))
    ])

    # Start with hog probabilities, then add pdm, then add landmarks_3d, then add embedded and finally facs
    #models = ['hog', 'pdm', 'landmarks_3d', 'embedded', 'facs', 'concat']
    #models = ['hog', 'pdm', 'landmarks_3d', 'embeddings', 'facs','concat']
    # Take the important ones
    models = ['hog', 'pdm', 'embeddings', 'facs']


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

    #do_experiment_subset()



    X_stack = np.concatenate([probabilities_val[model] for model in models], axis=1)
    X_stack_test = np.concatenate([probabilities_test[model] for model in models], axis=1)
    stacking_pipeline.fit(X_stack, y_val)
    explainer = shap.Explainer(stacking_pipeline.named_steps['log_reg'], X_stack)

    # Only look at the correctly classified samples
    correct_indices = np.where(stacking_pipeline.predict(X_stack_test) == y_test)[0]

    X_stack_test = X_stack_test[correct_indices]
    y_test = y_test[correct_indices]

    shap_values_test = explainer(X_stack_test)

    joblib.dump(shap_values_test, f'SV_test_correct.joblib')

    #shap.bar_plot(shap_values_test, max_display=10)


    logger.info("Experiment Finished")