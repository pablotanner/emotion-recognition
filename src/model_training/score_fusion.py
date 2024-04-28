import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
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

    # First Give individual accuracies
    for model in models:
        print(f"{model.__class__.__name__} accuracy: {model.score(X_test, y_test)}")

    # Then give Evaluation
    evaluate_results(y_test, final_predictions)

