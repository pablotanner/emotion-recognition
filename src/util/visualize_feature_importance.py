import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def show_feature_importance(X_train, y_train, X):
    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)

    # Get feature importance
    random_forest.fit(X_train, y_train)

    feature_importances = random_forest.feature_importances_

    # Get the indices of the most important features
    start_time = time.time()
    std = np.std([tree.feature_importances_ for tree in random_forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    feature_names = [f"feature {i}" for i in range(X.shape[1])]

    forest_importances = pd.Series(feature_importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
