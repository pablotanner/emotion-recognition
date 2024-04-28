import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def select_important_features(X, y, n_top_features=50):
    # Initialize AdaBoost with decision stumps
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )

    # Fit AdaBoost model
    ada.fit(X, y)

    # Extract feature importances
    importances = np.mean([tree.feature_importances_ for tree in ada.estimators_], axis=0)

    # Sort features by importance
    top_features_indices = np.argsort(importances)[::-1][:n_top_features]

    # Reduce X to top features
    X_reduced = X[:, top_features_indices]

    return X_reduced #, top_features_indices