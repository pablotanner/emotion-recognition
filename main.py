import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from model_explanation import select_features, get_important_features
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, StandardScalerStrategy
from src.evaluation.evaluate import evaluate_results
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM, MLP, RandomForestModel



data_loader = DataLoader("./data", "./data")

#feature_scores = np.load('feature_scores_int_nonrigid_3d.npy', allow_pickle=True).item()

feature_fuser = FeatureFuser(
    data_loader.features,
    include=['landmark_distances'],
    fusion_strategy=CompositeFusionStrategy([StandardScalerStrategy()])
)

# Has best results with 35 threshold (46.43% with SVM Linear)
#important_feature_names = get_important_features(path='feature_scores_int_nonrigid_3d.npy', threshold=35)

y = data_loader.emotions

# If important_feature_names is provided, only the features from the list will be used
X = feature_fuser.get_fused_features()  # important_feature_names=important_feature_names)

data_splitter = DataSplitter(X, y, test_size=0.2)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = data_splitter.split_data()

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

feature_scores = select_features(svm, feature_fuser.feature_names, X_train, y_train, X_test, y_test)
