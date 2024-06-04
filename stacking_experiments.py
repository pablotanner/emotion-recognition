import argparse
import logging
from cuml.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight
from classifier_vs_feature_experiment import get_tuned_classifiers
from src.util.data_paths import get_data_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Score Fusion Stacking Experiments')
    parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                        default='/local/scratch/ptanner/stacking_experiments')
    #parser.add_argument('--feature', type=str, help='Feature to use for training', default='pdm')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/experiment.log'),
                            logging.StreamHandler()
                        ])

    logger.info("Starting Experiment")

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    single_feature_results = {feature: {} for feature in ['nonrigid_face_shape','hog','landmarks_3d','facs', 'embedded']}
    single_classifier_results = {clf_name: {} for clf_name in ['LogisticRegression', 'NN', 'SVC', 'MLP', 'LinearSVC', 'RandomForest']}

    features = ['nonrigid_face_shape','hog','landmarks_3d','facs', 'embedded']
    classifier_names = ['LogisticRegression', 'NN', 'SVC', 'MLP', 'LinearSVC', 'RandomForest']

    predicted_probabilities_val = {clf_name: {feature: None for feature in features} for clf_name in classifier_names}
    predicted_probabilities_test = {clf_name: {feature: None for feature in features} for clf_name in classifier_names}
    def single_feature_experiment(feature):
        logger.info(f"Training on feature: {feature}")
        # Load data
        train_data = np.load(get_data_path('train', feature)).astype(np.float32)
        val_data = np.load(get_data_path('val', feature)).astype(np.float32)
        test_data = np.load(get_data_path('test', feature)).astype(np.float32)

        probabilities_val = {}
        probabilities_test = {}

        classifier_dict = get_tuned_classifiers(feature, class_weights, test_data.shape[1])


        # Train and evaluate classifiers
        for clf_name, classifier in classifier_dict.items():
            logger.info(f"Training {clf_name} on {feature}")
            classifier.fit(train_data, y_train)
            proba_val = classifier.predict_proba(val_data)
            proba_test = classifier.predict_proba(test_data)
            bal_acc_val = balanced_accuracy_score(y_val, np.argmax(proba_val, axis=1))
            bal_acc_test = balanced_accuracy_score(y_test, np.argmax(proba_test, axis=1))
            logger.info(f"Balanced Accuracy {clf_name}: {bal_acc_val} / {bal_acc_test}")
            single_feature_results[feature][clf_name] = bal_acc_test

            probabilities_val[clf_name] = proba_val
            probabilities_test[clf_name] = proba_test

            predicted_probabilities_val[clf_name][feature] = proba_val
            predicted_probabilities_test[clf_name][feature] = proba_test

            # Reset Model to save memory
            classifier_dict[clf_name] = 'Done'

        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1)

        stacking_pipeline = Pipeline(
            [('scaler', StandardScaler()), ('log_reg', LogisticRegression(C=1, class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

        bal_acc_stack = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_test_stack))

        logger.info(f"Stacking Performance: {stacking_pipeline.score(X_stack, y_val)}/{bal_acc_stack}")

        single_feature_results[feature]['Stacking'] = bal_acc_stack


    for feature in features:
        single_feature_experiment(feature)
        logger.info(f"Finished Experiment for {feature}")

    np.save(f'{args.experiment_dir}/single_feature_results.npy', single_feature_results)

    def single_classifier_experiment(clf_name):
        logger.info(f"Training on classifier: {clf_name}")
        probabilities_val = {}
        probabilities_test = {}

        for feature in features:
            logger.info(f"Training on {feature}")

            # Check if predictions have already been made for this feature-classifier pair
            if predicted_probabilities_val[clf_name][feature] is not None:
                probabilities_val[feature] = predicted_probabilities_val[clf_name][feature]
                probabilities_test[feature] = predicted_probabilities_test[clf_name][feature]

                logger.info(f"Found predictions for {feature}, skipping...")
                logger.info(f"Balanced Accuracy {feature}: {balanced_accuracy_score(y_val, np.argmax(probabilities_val[feature], axis=1))} / {balanced_accuracy_score(y_test, np.argmax(probabilities_test[feature], axis=1))}")
                continue

            train_data = np.load(get_data_path('train', feature)).astype(np.float32)
            val_data = np.load(get_data_path('val', feature)).astype(np.float32)
            test_data = np.load(get_data_path('test', feature)).astype(np.float32)

            classifier = get_tuned_classifiers(feature, class_weights, test_data.shape[1])[clf_name]

            classifier.fit(train_data, y_train)
            proba_val = classifier.predict_proba(val_data)
            proba_test = classifier.predict_proba(test_data)
            bal_acc_val = balanced_accuracy_score(y_val, np.argmax(proba_val, axis=1))
            bal_acc_test = balanced_accuracy_score(y_test, np.argmax(proba_test, axis=1))
            logger.info(f"Balanced Accuracy {feature}: {bal_acc_val} / {bal_acc_test}")
            single_classifier_results[clf_name][feature] = bal_acc_test

            probabilities_val[feature] = proba_val
            probabilities_test[feature] = proba_test

            predicted_probabilities_val[clf_name][feature] = proba_val
            predicted_probabilities_test[clf_name][feature] = proba_test


        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1)

        stacking_pipeline = Pipeline(
            [('scaler', StandardScaler()), ('log_reg', LogisticRegression(C=1, class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

        bal_acc_stack = balanced_accuracy_score(y_test, stacking_pipeline.predict(X_test_stack))

        logger.info(f"Stacking Performance: {stacking_pipeline.score(X_stack, y_val)}/{bal_acc_stack}")

        single_classifier_results[clf_name]['Stacking'] = bal_acc_stack

    for clf_name in classifier_names:
        single_classifier_experiment(clf_name)
        logger.info(f"Finished Experiment for {clf_name}")

    np.save(f'{args.experiment_dir}/single_classifier_results.npy', single_classifier_results)

    np.save(f'{args.experiment_dir}/predicted_probabilities_val.npy', predicted_probabilities_val)
    np.save(f'{args.experiment_dir}/predicted_probabilities_test.npy', predicted_probabilities_test)

    logger.info("Experiment Finished")