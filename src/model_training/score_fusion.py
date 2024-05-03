import copy

import numpy as np
from numpy import ndarray
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorflow.python.keras.utils.np_utils import to_categorical

from src import evaluate_results

default_models = [
    RandomForestClassifier(n_estimators=200, random_state=42),
    SVC(kernel='linear', probability=True, random_state=42),
    MLPClassifier(hidden_layer_sizes=(200,), random_state=42)
]

def perform_score_fusion_basic(X_train, X_test, y_train, y_test, models=default_models):
    for model in models:
        model.fit(X_train, y_train)

    probabilities = [model.predict_proba(X_test) for model in models]
    average_probs = np.mean(probabilities, axis=0)
    final_predictions: ndarray[int] = np.argmax(average_probs, axis=1)

    # For each model, print the balanced accuracy
    for model in models:
        y_pred = model.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"{model.__class__.__name__} balanced accuracy: {bal_acc}")
    # Then give Evaluation
    evaluate_results(y_test, final_predictions)


def perform_score_fusion(X, y, models, technique='average', n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    overall_accuracies = []
    individual_accuracies = {model.__name__: [] for model in models}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        original_y_test = y_test.copy()  # Keep an unmodified copy for evaluation


        probabilities = []
        predictions = []


        for model in models:
            # If model is a neural network, it means we passed in a function to create the model
            if model.__class__.__name__ == 'function':
                # Create copy
                model = model(X_train.shape[1])

                # Handle neural network training
                y_train_nn = to_categorical(y_train, num_classes=8)
                model.fit(X_train, y_train_nn, epochs=50, batch_size=32, verbose=0)
                proba = model.predict(X_test)
                pred = np.argmax(proba, axis=1)
            else:
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)
                pred = model.predict(X_test)

            probabilities.append(proba)
            predictions.append(pred)

            print(f"{model.__class__.__name__} balanced accuracy: {balanced_accuracy_score(y_test, pred)}")
            evaluate_results(y_test, pred)

            # Evaluate individual model performance
            acc = balanced_accuracy_score(original_y_test, pred)
            individual_accuracies[model.__name__].append(acc)

        # Evaluate fusion techniques
        if technique == 'average':
            mean_probs = np.mean(probabilities, axis=0)
            final_predictions = np.argmax(mean_probs, axis=1)
        elif technique == 'majority_vote':
            final_predictions = mode(predictions, axis=0)[0].flatten()
        elif technique == 'stacking':
            # Ensure all models are retrained within the stack
            stack = StackingClassifier(estimators=[(model.__name__, model) for model in models],
                                       final_estimator=LogisticRegression(), cv=3)
            stack.fit(X_train, y_train)
            final_predictions = stack.predict(X_test)



        # Evaluate results
        acc = balanced_accuracy_score(original_y_test, final_predictions)
        overall_accuracies.append(acc)

        # evaluate_results(original_y_test, final_predictions)

    # Print individual
    for model, accuracies in individual_accuracies.items():
        print(f"{model} - Average Balanced Accuracy: {np.mean(accuracies)}")
    # Print overall
    print(f"Technique {technique} - Average Balanced Accuracy: {np.mean(overall_accuracies)}")
