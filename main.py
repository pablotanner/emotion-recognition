import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from src.feature_importance.model_explanation import select_features, get_important_features
from src.data_processing.data_loader import DataLoader
from src.data_processing.feature_fuser import FeatureFuser, CompositeFusionStrategy, StandardScalerStrategy
from src.evaluation.evaluate import evaluate_results
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM, MLP, RandomForestModel

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

data_loader = DataLoader("./data", "./data")


feature_fuser = FeatureFuser(
    data_loader.features,
    include=['nonrigid_face_shape', 'landmarks_3d', 'facs_intensity'],
    fusion_strategy=CompositeFusionStrategy([StandardScalerStrategy()])
)

# Has best results with 35 threshold (46.43% with SVM Linear)
# important_feature_names = get_important_features(path='feature_scores_new.npy', threshold=50)

y = data_loader.emotions

# If important_feature_names is provided, only the features from the list will be used
X = feature_fuser.get_fused_features() # important_feature_names=important_feature_names)

# Selecting the best features
X = SelectKBest(f_classif, k=100).fit_transform(X, y)

data_splitter = DataSplitter(X, y, test_size=0.2)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = data_splitter.split_data()


#svm = SVM(kernel='linear')
#svm.train(X_train, y_train)
#svm.evaluate(X_test, y_test)


svm = SVC(kernel='linear')