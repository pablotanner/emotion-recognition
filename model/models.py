import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from model.evaluate import evaluate_results
from joblib import dump, load
import os

class BaseModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        evaluate_results(y_test, self.predict(X_test))

    def save_model(self, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f"store\\{name}.joblib")
        dump(self.model, filename)

    def load_model(self, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f"store\\{name}.joblib")
        self.model = load(filename)

class SVM(BaseModel):
    def __init__(self, C=1.0, kernel='rbf'):
        super().__init__()
        self.model = SVC(C=C, kernel=kernel)


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    @staticmethod
    # Find optimal number of trees for Random Forest (no scaler: 300, with scaler: 200, has 0.42 atm.)
    def tree_performance(X_train, X_test, y_train, y_test):
        test_scores = []

        n_estimators_options = [50, 100, 200, 300, 400, 500]
        for n_estimators in n_estimators_options:
            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            clf.fit(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            test_scores.append(test_score)

        plt.plot(n_estimators_options, test_scores)
        plt.xlabel("Number of Trees")
        plt.ylabel("Accuracy")
        plt.title("Performance vs Number of Trees")
        plt.show()