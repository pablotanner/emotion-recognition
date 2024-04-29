import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, VarianceThreshold, f_classif
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.feature_importance.feature_selection import select_important_features
from src.feature_importance.model_explanation import select_features, get_important_features
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, StandardScalerStrategy
from src.evaluation.evaluate import evaluate_results
from src.model_training.data_splitter import DataSplitter

from src.model_training.score_fusion import perform_score_fusion

data_loader = DataLoader("./data", "./data")


feature_fuser = FeatureFuser(
    data_loader.features,
    include=['nonrigid_face_shape', 'landmarks_3d', 'facs_intensity'],
    #fusion_strategy=CompositeFusionStrategy([StandardScalerStrategy()])
)
y = data_loader.emotions

# If important_feature_names is provided, only the features from the list will be used
X = feature_fuser.get_fused_features()

def no_selection_experiment(X, y):
    # Splitting data first
    data_splitter = DataSplitter(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Use StandardScaler to scale the data
    X_train, X_test = data_splitter.scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test

"""
Experiments using AdaBoost for feature selection, first selects top n features, then trains models
"""
def adaboost_experiment(X, y):
    X_selected, top_indices = select_important_features(X, y, n_top_features=50)
    top_feature_names = [feature_fuser.feature_names[i] for i in top_indices]

    data_splitter = DataSplitter(X_selected, y, test_size=0.2)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Use StandardScaler to scale the data
    X_train, X_test = data_splitter.scale_data(X_train, X_test)

    return X_train, X_test, y_train, y_test


"""
Experiments where Feature Selection uses Filter approach, meaning that the methods are applied
before the model training.
"""
def filter_experiment(X, y):
    # Split data first
    data_splitter = DataSplitter(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Use StandardScaler to scale the data
    X_train, X_test = data_splitter.scale_data(X_train, X_test)

    # Apply feature selection based on the training data
    selector = SelectKBest(f_classif, k=50)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, y_train, y_test

"""
Experiments where Feature Selection uses Wrapper approach, meaning that feature subsets are evaluated
based on model performance.
"""
def wrapper_experiment(X, y):
    # Splitting data first
    data_splitter = DataSplitter(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Wrapper method applied with training data
    selector = RFE(RandomForestClassifier(), n_features_to_select=50)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, y_train, y_test

"""
Experiments where Feature Selection uses Embedded approach, meaning that feature selection is done
during the model training itself.
"""
def embedded_experiment(X, y):
    data_splitter = DataSplitter(X, y, test_size=0.2)
    # Split data first
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    # Scale data
    X_train, X_test = data_splitter.scale_data(X_train, X_test)


    models = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        #Lasso(alpha=0.01),
        SVC(kernel='linear', C=1.0)
    ]

    for model in models:
        model.fit(X_train, y_train)
        importances = get_feature_importances(model)
        if importances is not None:
            top_indices = np.argsort(importances)[::-1][:50]
            #top_feature_names = [feature_names[i] for i in top_indices]

            X_train_selected = X_train[:, top_indices]
            X_test_selected = X_test[:, top_indices]

            # Re-train model on selected features
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy of {model.__class__.__name__}: {accuracy:.2f}')

def get_feature_importances(model):
    try:
        # Attempt to retrieve feature importances directly
        importances = model.feature_importances_
    except AttributeError:
        # For models that do not support `feature_importances_` directly
        if hasattr(model, 'coef_'):
            # Handle models with coefficients (like linear models)
            importances = np.abs(model.coef_[0])
        else:
            # Set a default or handle other cases
            importances = None
    return importances






X_train, X_test, y_train, y_test = None, None, None, None

EXPERIMENT = "wrapper"

if EXPERIMENT == "no_selection":
    X_train, X_test, y_train, y_test = no_selection_experiment(X, y)
elif EXPERIMENT == "adaboost":
    X_train, X_test, y_train, y_test = adaboost_experiment(X, y)
elif EXPERIMENT == "filter":
    X_train, X_test, y_train, y_test = filter_experiment(X, y)
elif EXPERIMENT == "wrapper":
    X_train, X_test, y_train, y_test = wrapper_experiment(X, y)
elif EXPERIMENT == "embedded":
    embedded_experiment(X, y)
    exit()

"""
If experiment isn't the embedded one, it means we have to initialize and train models ourselves
"""
if EXPERIMENT != "embedded":
    # Initialize models using parameters from grid search (grid_search_results.npy)
    svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), solver='sgd', learning_rate_init=0.001, activation='tanh',
                        random_state=42)
    models = [svm, mlp, rf]

    print(20 * "=" + f" {EXPERIMENT} " + 20 * "=")
    # Perform score fusion
    perform_score_fusion(X_train, X_test, y_train, y_test, models=models)




# Perform grid search
# results = run_grid_search(X_train, y_train)

