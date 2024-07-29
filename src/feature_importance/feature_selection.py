import numpy as np
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


"""
Feature selection methods, were not used in the final thesis
"""

def select_embedded_adaboost(X, y, n_top_features=50):
    clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, prefit=True, max_features=n_top_features, threshold=-np.inf)

    X_new = model.transform(X)


    return X_new

def select_features_adaboost(X, y, n_top_features=50):
    # Initialize AdaBoost with decision stumps
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Fit AdaBoost model
    ada.fit(X, y)

    # Extract feature importances
    importances = np.mean([tree.feature_importances_ for tree in ada.estimators_], axis=0)

    # Sort features by importance_visualizations
    top_features_indices = np.argsort(importances)[::-1][:n_top_features]

    # Reduce X to top features
    X_reduced = X[:, top_features_indices]

    return X_reduced, top_features_indices

def select_features_adaboost_new(X_train, X_test, y, n_top_features=50):
    # Initialize AdaBoost with decision stumps
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )

    # Fit AdaBoost model
    ada.fit(X_train, y)


    # Extract feature importances
    importances = np.mean([tree.feature_importances_ for tree in ada.estimators_], axis=0)

    # Sort features by importance_visualizations
    top_features_indices = np.argsort(importances)[::-1][:n_top_features]

    # Reduce X to top features
    X_train_reduced = X_train[:, top_features_indices]
    X_test_reduced = X_test[:, top_features_indices]

    return X_train_reduced, X_test_reduced, top_features_indices

def select_features_tree_based_before(X, y, max_features=50):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, prefit=True, max_features=max_features, threshold=-np.inf)

    X_new = model.transform(X)

    return X_new


# Embedded approach
def select_features_tree_based(X_train, X_test, y_train, max_features=50):
    clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
    clf = clf.fit(X_train, y_train)

    model = SelectFromModel(clf, prefit=True, max_features=max_features)

    X_train_new = model.transform(X_train)
    X_test_new = model.transform(X_test)

    return X_train_new, X_test_new


# Wrapper approach
def select_features_sequential(X, y, n_top_features=50, estimator=None, direction='forward'):
    if estimator is None:
        estimator = KNeighborsClassifier(n_neighbors=3)

    sfs = SequentialFeatureSelector(estimator, n_features_to_select=n_top_features, direction=direction)
    sfs.fit(X, y)

    # Return transformed data and indices of selected features
    return sfs.transform(X), sfs.get_support(indices=True)

# Wrapper approach
def select_features_rfe(X_train, X_test, y_train, n_top_features=50, estimator=None):
    estimator = RandomForestClassifier() if estimator is None else estimator

    selector = RFE(estimator, n_features_to_select=n_top_features, step=1)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected