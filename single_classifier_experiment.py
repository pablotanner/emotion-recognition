import argparse
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import compute_class_weight
import joblib
from classifier_vs_feature_experiment import feature_paths, get_tuned_classifiers, evaluate_stacking
import os




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training one classifier with all feature types')
    parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                        default='/local/scratch/ptanner/single_classifier_experiments')
    parser.add_argument('--classifier', type=str, help='Classifier to use', default='SVC')
    args = parser.parse_args()

    if args.classifier not in ['SVC', 'RandomForest', 'LogisticRegression', 'MLP', 'NN', 'LinearSVC']:
        raise ValueError(f"Classifier {args.classifier} not supported")

    features = ['hog', 'landmarks_3d', 'pdm', 'facs', 'embedded']

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{args.experiment_dir}/{args.classifier}.log'),
                            logging.StreamHandler()
                        ])
    logger.info("Starting Experiment")

    if not os.path.exists(f'{args.experiment_dir}/{args.classifier}'):
        os.makedirs(f'{args.experiment_dir}/{args.classifier}')

    probabilities_val = {}
    probabilities_test = {}

    # For each feature, train the clf with best params
    logger.info(f"Starting Training of {args.classifier}")
    for feature in features:
        if os.path.exists(f'{args.experiment_dir}/{args.classifier}/{feature}.joblib'):
            logger.info(f"Found {feature} features already trained, loading model...")
            clf = joblib.load(f'{args.experiment_dir}/{args.classifier}/{feature}.joblib')
        else:
            logger.info(f"Training on {feature} features")
            X_shape = np.load(feature_paths[feature]['test']).shape[1]
            clf = get_tuned_classifiers(feature, class_weights, X_shape)[args.classifier]
            clf.fit(np.load(feature_paths[feature]['train']), y_train)
            logger.info(f"Training Complete on {feature} features")
            joblib.dump(clf, f'{args.experiment_dir}/{args.classifier}/{feature}.joblib')
            logger.info(f"Model saved for {feature} features")
        probabilities_val[feature] = clf.predict_proba(np.load(feature_paths[feature]['val']))
        probabilities_test[feature] = clf.predict_proba(np.load(feature_paths[feature]['test']))
        bal_acc_val = balanced_accuracy_score(y_val, np.argmax(probabilities_val[feature], axis=1))
        bal_acc_test = balanced_accuracy_score(y_test, np.argmax(probabilities_test[feature], axis=1))
        logger.info(f"[{feature}] Balanced Accuracy on Validation: {bal_acc_val}")
        logger.info(f"[{feature}] Balanced Accuracy on Test: {bal_acc_test}")

    logger.info(f"---------------------{args.classifier} Stacking Results---------------------")
    stacking_pipeline = evaluate_stacking(probabilities_val, y_val)

    # Evaluate the stacking classifier on the test set
    X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)
    test_accuracy = stacking_pipeline.score(X_test_stack, y_test)
    logger.info(f"Accuracy of stacking classifier (Test Set): {test_accuracy}")

    logger.info("Experiment Finished")



