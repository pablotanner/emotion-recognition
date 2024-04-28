import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.feature_importance.feature_selection import select_important_features
from src.feature_importance.model_explanation import select_features, get_important_features
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, StandardScalerStrategy
from src.evaluation.evaluate import evaluate_results
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM, MLP, RandomForestModel
from src.model_training.grid_search import run_grid_search
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

X = select_important_features(X, y, n_top_features=50)

data_splitter = DataSplitter(X, y, test_size=0.2)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = data_splitter.split_data()

# Use StandardScaler to scale the data
X_train, X_test = data_splitter.scale_data(X_train, X_test)

# Perform grid search
#results = run_grid_search(X_train, y_train)

# Initialize models using parameters from grid search (grid_search_results.npy)
svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True)
rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50),solver='sgd', learning_rate_init=0.001, activation='tanh')
models = [svm, mlp]

# Perform score fusion
perform_score_fusion(X_train, X_test, y_train, y_test, models=models)


