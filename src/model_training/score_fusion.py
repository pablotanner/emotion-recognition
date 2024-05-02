import numpy as np
from numpy import ndarray
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import evaluate_results

default_models = [
    RandomForestClassifier(n_estimators=200, random_state=42),
    SVC(kernel='linear', probability=True, random_state=42),
    MLPClassifier(hidden_layer_sizes=(200,), random_state=42)
]

def perform_score_fusion(X_train, X_test, y_train, y_test, models=default_models):
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


def perform_score_fusion_new(X, y, models, technique='average', n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    overall_accuracies = []

    model_predictions = {model: [] for model in models}
    model_accuracies = {model: [] for model in models}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        probabilities = []
        predictions = []


        for model in models:
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            pred = model.predict(X_test)
            probabilities.append(proba)
            predictions.append(pred)

            # For each model, store the predictions
            model_predictions[model].extend(pred)


        # For each model, store the balanced accuracy
        for model in models:
            y_pred = model.predict(X_test)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            model_accuracies[model].append(bal_acc)

        if technique == 'average':
            # Compute average of probabilities
            mean_probs = np.mean(probabilities, axis=0)
            final_predictions = np.argmax(mean_probs, axis=1)
        elif technique == 'majority_vote':
            # Compute majority vote
            final_predictions = mode(predictions, axis=0)[0].flatten()
        elif technique == 'stacking':
            # Stacking classifier
            stack = StackingClassifier(estimators=[(model.__class__.__name__, model) for model in models],
                                       final_estimator=LogisticRegression(), cv=3)
            stack.fit(X_train, y_train)
            final_predictions = stack.predict(X_test)
        else:
            raise ValueError("Unsupported fusion technique specified")

        # Evaluate results
        acc = balanced_accuracy_score(y_test, final_predictions)
        overall_accuracies.append(acc)

    # For each model, print the balanced accuracy
    for model in models:
        bal_acc = np.mean(model_accuracies[model])
        print(f"{model.__class__.__name__} balanced accuracy: {bal_acc}")
    print(f"Technique {technique} - Average Balanced Accuracy: {np.mean(overall_accuracies)}")
