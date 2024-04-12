import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os

from src.evaluation.evaluate import evaluate_results


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
        filename = os.path.join(dirname, f"../../models/store/{name}.joblib")
        dump(self.model, filename)

    def load_model(self, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, f"../../models/store/{name}.joblib")
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

class MLP(BaseModel):
    def __init__(self, hidden_layer_sizes=(200,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200):
        super().__init__()
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter)

    # Currently best performance (accuracy) at 200 hidden layers
    @staticmethod
    def mlp_performance(X_train, X_test, y_train, y_test):
        test_scores = []

        hidden_layer_sizes_options = [(100,), (200,), (300,), (400,), (500,)]
        for hidden_layer_sizes in hidden_layer_sizes_options:
            clf = MLPClassifier(solver="sgd",hidden_layer_sizes=hidden_layer_sizes, max_iter=200, random_state=42)
            clf.fit(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            test_scores.append(test_score)

        plt.plot(hidden_layer_sizes_options, test_scores)
        plt.xlabel("Hidden Layer Sizes")
        plt.ylabel("Accuracy")
        plt.title("Performance vs Hidden Layer Sizes")
        plt.show()
