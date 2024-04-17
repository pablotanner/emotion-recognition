from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, KernelTransformerStrategy, StandardScalerStrategy
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM, MLP, RandomForestModel
from src.util.visualize_feature_importance import show_feature_importance

data_loader = DataLoader("./data")

feature_fuser = FeatureFuser(
    data_loader.features,
    #include=['facs_intensity','landmarks', 'nonrigid_face_shape'],
    include=['facs_intensity','landmark_distances', 'facs_presence'],
    fusion_strategy=CompositeFusionStrategy([StandardScalerStrategy()])
)

y = data_loader.emotions
X = feature_fuser.fuse_features()


data_splitter = DataSplitter(X, y, test_size=0.2)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = data_splitter.split_data()



"""
Initializes Models, in general I try to save the models trained with FACS presence and intensity features 
(great accuracy with few features)
"""
def initialize_models():
    # For each models, check if it already exists in the store, if not, train it and save it
    svm = SVM(C=1.0, kernel='linear')
    mlp = MLP(solver='sgd', max_iter=1000, hidden_layer_sizes=(400,))
    random_forest = RandomForestModel(n_estimators=200, max_depth=10)

    try:
        svm.load_model('svm-linear')
    except FileNotFoundError:
        svm.train(X_train, y_train)
        svm.save_model('svm-linear')

    try:
        random_forest.load_model('random_forest')
    except FileNotFoundError:
        random_forest.train(X_train, y_train)
        random_forest.save_model('random_forest')

    try:
        mlp.load_model('mlp-sgd')
    except FileNotFoundError:
        mlp.train(X_train, y_train)
        mlp.save_model('mlp-sgd')

    return svm, random_forest, mlp


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

"""
When only using FACS Features, SGD is  best with 41.76% 
"""
def compare_mlp_solvers(hidden_layer_sizes=(400,)):
    # Init the models
    mlp_adam = MLP(solver='adam', hidden_layer_sizes=hidden_layer_sizes)
    mlp_lbfgs = MLP(solver='lbfgs', hidden_layer_sizes=hidden_layer_sizes)
    mlp_sgd = MLP(solver='sgd', hidden_layer_sizes=hidden_layer_sizes)

    # Train the models
    mlp_adam.train(X_train, y_train)
    mlp_lbfgs.train(X_train, y_train)
    mlp_sgd.train(X_train, y_train)

    # Evaluate the models
    print("MLP Adam:")
    mlp_adam.evaluate(X_test, y_test)

    print("MLP LBFGS:")
    mlp_lbfgs.evaluate(X_test, y_test)

    print("MLP SGD:")
    mlp_sgd.evaluate(X_test, y_test)

"""
Boosted Random Forest

base_rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
adaboost_rf_clf = AdaBoostClassifier(base_estimator=base_rf_clf, n_estimators=50, random_state=42)

adaboost_rf_clf.fit(X_train, y_train)

y_pred = adaboost_rf_clf.predict(X_test)

evaluate_results(y_test, y_pred)

"""

svm = SVM(C=1.0, kernel='linear')

svm.train(X_train, y_train)

svm.evaluate(X_test, y_test)


