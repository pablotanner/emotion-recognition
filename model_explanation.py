import lime
import numpy as np
from lime import lime_tabular
import math


def createExplainer(X_train, y_train, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train, training_labels=y_train,
                                                       class_names=class_names, feature_names=feature_names)
    return explainer


def explain_index(index, explainer, model, X_test, y_test, class_names, show=False):
    sample = X_test[index]
    explanation = explainer.explain_instance(sample, model.predict_proba, num_features=10, top_labels=7)
    predicted_index = model.predict(X_test)[index]
    if show:
        explanation.show_in_notebook()
        # Expected vs True
        print("Expected: ", class_names[y_test[index]])
        predicted_index = model.predict(X_test)[index]
        print("Predicted: ", class_names[predicted_index])
    return explanation, predicted_index


def extract_important_features(explanation, predicted_index):
    # Given as " ('facs_intensity_8 <= 0.00', 0.026830424167673558)", we need to get out facs_intensity_8
    feature_list = explanation.as_list(label=predicted_index)

    def contains_underline(text):
        return '_' in text

    # Get the feature names (the one with underline)
    feature_scores = {}

    for feature, value in feature_list:
        feature_name = list(filter(contains_underline, feature.split()))[0]
        value = math.ceil(1000*value)
        feature_scores[feature_name] = value

    return feature_scores


def select_features(trained_model, feature_names, X_train, y_train, X_test, y_test):
    class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

    # Initialize the explainer
    explainer = createExplainer(X_train, y_train, feature_names, class_names)

    feature_scores = {}

    for i in range(len(X_test)):
        explanation, predicted_index = explain_index(i, explainer, trained_model, X_test, y_test, class_names, show=False)
        features = extract_important_features(explanation, predicted_index)

        print(f"Feature Scores {i}: {feature_scores}")

        for feature, score in features.items():
            if feature in feature_scores:
                feature_scores[feature] += score
            else:
                feature_scores[feature] = score
    return feature_scores


def get_important_features(path, threshold=0):
    feature_scores = np.load(path, allow_pickle=True).item()
    important_features = [feature for feature, score in feature_scores.items() if score >= threshold]
    return important_features
