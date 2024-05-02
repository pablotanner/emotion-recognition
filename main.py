import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, VarianceThreshold, f_classif, chi2, \
    mutual_info_classif, SelectFpr, f_regression, SelectFwe
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from src.feature_importance.feature_selection import select_features_adaboost, select_features_tree_based, \
    select_features_rfe, select_features_sequential, select_features_adaboost_new, select_embedded_adaboost, \
    select_features_tree_based_before
from src.feature_importance.model_explanation import select_features, get_important_features
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, StandardScalerStrategy
from src.evaluation.evaluate import evaluate_results
from src.model_training.data_splitter import DataSplitter
from src.model_training.grid_search import run_grid_search

from src.model_training.score_fusion import perform_score_fusion, perform_score_fusion_new

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
def adaboost_experiment(X, y, variant="before"):
    if variant == "before":
        X_selected, top_indices = select_features_adaboost(X, y, n_top_features=40)
        top_feature_names = [feature_fuser.feature_names[i] for i in top_indices]

        data_splitter = DataSplitter(X_selected, y, test_size=0.2)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = data_splitter.split_data()

        # Use StandardScaler to scale the data
        X_train, X_test = data_splitter.scale_data(X_train, X_test)

        return X_train, X_test, y_train, y_test
    elif variant == "after":
        data_splitter = DataSplitter(X, y, test_size=0.2)
        # Split data first
        X_train, X_test, y_train, y_test = data_splitter.split_data()

        X_train, X_test = data_splitter.scale_data(X_train, X_test)

        X_train, X_test, top_indices = select_features_adaboost_new(X_train, X_test, y_train, n_top_features=50)

        return X_train, X_test, y_train, y_test
    elif variant == "embedded":
        X_selected = select_embedded_adaboost(X, y, n_top_features=120)

        data_splitter = DataSplitter(X_selected, y, test_size=0.2)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = data_splitter.split_data()

        # Use StandardScaler to scale the data
        X_train, X_test = data_splitter.scale_data(X_train, X_test)

        return X_train, X_test, y_train, y_test
    else:
        raise ValueError("Invalid variant")


"""
Experiments where Feature Selection uses Filter approach, meaning that the methods are applied
before the model training.
"""
def filter_experiment(X, y, selection_method="f_classif"):
    k_features = 200
    # Split data first
    data_splitter = DataSplitter(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = data_splitter.split_data()

    if selection_method == "chi2":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Use scaler to scale the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Select the feature selection strategy
    if selection_method == 'chi2':
        selector = SelectKBest(chi2, k=k_features)
    elif selection_method == 'f_classif':
        selector = SelectKBest(f_classif, k=k_features)
    elif selection_method == 'mutual_info_classif':
        selector = SelectKBest(mutual_info_classif, k=k_features)
    elif selection_method == 'SelectFpr':
        # Using f_regression for a continuous outcome example, replace as needed
        selector = SelectFpr(f_regression, alpha=0.05)  # Adjust alpha as necessary
    else:
        raise ValueError("Unsupported selection method")

    # Apply feature selection based on the training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    return X_train_selected, X_test_selected, y_train, y_test

"""
Experiments where Feature Selection uses Wrapper approach, meaning that feature subsets are evaluated
based on model performance.
"""
def wrapper_experiment(X, y, transformer="rfe"):
    # Splitting data first
    data_splitter = DataSplitter(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = data_splitter.split_data()
    X_train, X_test = data_splitter.scale_data(X_train, X_test)

    if transformer == "rfe":
        X_train, X_test = select_features_rfe(X_train, X_test, y_train, n_top_features=50)
    elif transformer == "sequential":
        _, indices = select_features_sequential(X_train, y_train, n_top_features=50)
        X_train = X_train[:, indices]
        X_test = X_test[:, indices]
    else:
        raise ValueError("Invalid transformer")
    print(20 * "*" + f" {transformer} " + 20 * "*")
    return X_train, X_test, y_train, y_test





"""
Experiments where Feature Selection uses Embedded approach, meaning that feature selection is done
during the model training itself.
"""
def embedded_experiment(X, y, variant="after"):
    if variant == "before":
        X = select_features_tree_based_before(X, y, max_features=100)

        data_splitter = DataSplitter(X, y, test_size=0.2)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = data_splitter.split_data()

        # Use StandardScaler to scale the data
        X_train, X_test = data_splitter.scale_data(X_train, X_test)

        return X_train, X_test, y_train, y_test

    elif variant == "after":
        data_splitter = DataSplitter(X, y, test_size=0.2)
        # Split data first
        X_train, X_test, y_train, y_test = data_splitter.split_data()

        X_train, X_test = data_splitter.scale_data(X_train, X_test)

        X_train, X_test = select_features_tree_based(X_train, X_test, y_train, max_features=50)

        return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = None, None, None, None


start_time = time.time()


EXPERIMENT = "adaboost"

if EXPERIMENT == "no_selection":
    X_train, X_test, y_train, y_test = no_selection_experiment(X, y)
elif EXPERIMENT == "adaboost":
    X_train, X_test, y_train, y_test = adaboost_experiment(X, y, variant="before")
elif EXPERIMENT == "filter":
    X_train, X_test, y_train, y_test = filter_experiment(X, y, selection_method="f_classif")
elif EXPERIMENT == "wrapper":
    X_train, X_test, y_train, y_test = wrapper_experiment(X, y, transformer="rfe")
elif EXPERIMENT == "embedded":
    X_train, X_test, y_train, y_test = embedded_experiment(X, y, variant="before")


# Initialize models using parameters from grid search (grid_search_results.npy)
# NO HOG
svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='tanh', random_state=42)


# WITH HOG
#svm = SVC(C=0.1, gamma='scale', kernel='linear', probability=True, random_state=42)
#rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
#mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='relu', random_state=42)


models = [svm, mlp, rf]

print(20 * "=" + f" {EXPERIMENT} " + 20 * "=")

# Join the training and testing data
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Perform score fusion
#perform_score_fusion(X_train, X_test, y_train, y_test, models=models)
perform_score_fusion_new(X,y, models=models, n_splits=5, technique='average')


# Perform grid search
#results = run_grid_search(X_train, y_train)


print("--- %s seconds ---" % (time.time() - start_time))

