from model.models import RandomForestModel, SVM
from model.prepare_data import prepare_data, split_data

X, y = prepare_data(fake_features=False, use_affect_net_lnd=False, use_landmarks=False, use_facs_intensity=True, use_facs_presence=True)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y, use_scaler=True)


# Find optimal number of trees for Random Forest
# RandomForestModel.tree_performance(X_train, X_test, y_train, y_test)

def initialize_models():
    # For each model, check if it already exists in the store, if not, train it and save it
    svm = SVM(C=1.0, kernel='linear')
    random_forest = RandomForestModel(n_estimators=200, max_depth=10)

    try:
        svm.load_model('svm')
    except FileNotFoundError:
        svm.train(X_train, y_train)
        svm.save_model('svm')

    try:
        random_forest.load_model('random_forest')
    except FileNotFoundError:
        random_forest.train(X_train, y_train)
        random_forest.save_model('random_forest')

    return svm, random_forest


def compare_svm_rf(svm, random_forest):
    # Evaluate the models
    print("SVM:")
    svm.evaluate(X_test, y_test)

    print("Random Forest:")
    random_forest.evaluate(X_test, y_test)

"""
Linear is has best accuracy with 44.75%
"""
def compare_svm_kernels():
    # Init the models
    svm_linear = SVM(C=1.0, kernel='linear')
    svm_poly = SVM(C=1.0, kernel='poly')
    svm_rbf = SVM(C=1.0, kernel='rbf')

    # Train the models
    svm_linear.train(X_train, y_train)
    svm_poly.train(X_train, y_train)
    svm_rbf.train(X_train, y_train)

    # Evaluate the models
    print("SVM Linear:")
    svm_linear.evaluate(X_test, y_test)

    print("SVM Poly:")
    svm_poly.evaluate(X_test, y_test)

    print("SVM RBF:")
    svm_rbf.evaluate(X_test, y_test)

svm, random_forest = initialize_models()

