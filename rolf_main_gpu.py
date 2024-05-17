import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import cupy as cp
from cuml.svm import LinearSVC
from cuml.preprocessing import StandardScaler
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import compute_class_weight
from src.data_processing.rolf_loader import RolfLoader
import joblib

parser = argparse.ArgumentParser(description='Model training and evaluation (GPU)')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)')
args = parser.parse_args()

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, num_classes, num_epochs=10, batch_size=32, learning_rate=0.001,
                 class_weight=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.model = self._build_model()

        if self.class_weight is not None:
            weight_list = [self.class_weight[i] for i in range(self.num_classes)]
            weight_tensor = torch.tensor(weight_list, dtype=torch.float32).cuda()
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        return model

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_tensor = torch.tensor(y, dtype=torch.long).cuda()
        self.model = self.model.cuda()
        self.model.train()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return torch.softmax(outputs, dim=1).cpu().numpy()

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(message)s',
                        handlers=[
                            logging.FileHandler('logs/rolf_gpu_training.log'),
                            logging.StreamHandler()
                        ])
    logger.info("Loading data...")
    #data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir, args.test_features_dir, args.main_id_dir)


    logger.info("Data loaded.")

    #feature_splits_dict, emotions_splits_dict = data_loader.get_data()

    num_samples = 1000

    feature_splits_dict = {
        'train': {
            'landmarks_3d': np.random.rand(num_samples, 68 * 3),
            'facs_intensity': np.random.rand(num_samples, 20),
            'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
            'nonrigid_face_shape': np.random.rand(num_samples, 13),
            'hog': np.random.rand(num_samples, 3000)
        },
        'val': {
            'landmarks_3d': np.random.rand(num_samples, 68 * 3),
            'facs_intensity': np.random.rand(num_samples, 20),
            'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
            'nonrigid_face_shape': np.random.rand(num_samples, 13),
            'hog': np.random.rand(num_samples, 3000)
        },
        'test': {
            'landmarks_3d': np.random.rand(num_samples, 68 * 3),
            'facs_intensity': np.random.rand(num_samples, 20),
            'facs_presence': np.random.randint(0, 2, (num_samples, 20)),
            'nonrigid_face_shape': np.random.rand(num_samples, 13),
            'hog': np.random.rand(num_samples, 3000)
        },
    }
    # 8 Classes
    emotions_splits_dict = {
        'train': np.random.randint(0, 8, num_samples),
        'val': np.random.randint(0, 8, num_samples),
        'test': np.random.randint(0, 8, num_samples)
    }


    def evaluate_stacking(probabilities, y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression())
        ])

        stacking_pipeline.fit(X_stack, y_val)
        stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        logger.info(f"Accuracy of stacking classifier (Validation Set): {stacking_accuracy}")

        # Return the stacking pipeline
        return stacking_pipeline


    def save_features_to_disk(split_features_dict):
        """
        Save the features to disk
        """
        splits = list(split_features_dict.keys())

        for split in splits:
            np.save(f'{split}_spatial_features.npy', split_features_dict[split]['landmarks_3d'])
            np.save(f'{split}_facs_features.npy', np.hstack([split_features_dict[split]['facs_intensity'], split_features_dict[split]['facs_presence']]))
            np.save(f'{split}_pdm_features.npy', split_features_dict[split]['nonrigid_face_shape'])
            np.save(f'{split}_hog_features.npy', split_features_dict[split]['hog'])
            # Clear the dictionary to free up memory
            del split_features_dict[split]
            logger.info(f"Saved {split} features to disk")

    # Save features to disk and clear up from memory
    save_features_to_disk(feature_splits_dict)

    # Get the emotions for the train, validation, and test sets
    y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    def spatial_relationship_model(X, y):
        # Linear scores worse individually, but better in stacking
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=0.1, probability=True, class_weight=class_weights)) #  kernel='linear', gamma='scale'
        ])

        pipeline.fit(X, y)

        logger.info("Spatial Relationship Model Fitted")

        return pipeline


    def facial_unit_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10))
        ])

        pipeline.fit(X, y)

        logger.info("Facial Unit Model Fitted")

        return pipeline


    def pdm_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(C=0.1, solver='qn', class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("PDM Model Fitted")

        return pipeline


    def hog_model(X, y):
        n_components = 800  # Number of principal components to keep
        hidden_size = 200
        num_classes = len(np.unique(y))  # Number of classes

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components)),  # reduce dimensions
            ('mlp', PyTorchMLPClassifier(input_size=n_components, hidden_size=hidden_size, num_classes=num_classes,
                                         num_epochs=200, batch_size=32, learning_rate=0.001,
                                         class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("HOG Model Fitted")
        return pipeline

    logger.info("Starting Fitting...")


    probabilities_val = {}
    probabilities_test = {}

    # Train models, then save
    spatial_pipeline = spatial_relationship_model(np.load('train_spatial_features.npy'), y_train)
    probabilities_val["spatial"] = spatial_pipeline.predict_proba(np.load('val_spatial_features.npy'))
    probabilities_test["spatial"] = spatial_pipeline.predict_proba(np.load('test_spatial_features.npy'))
    # Log individual accuracy
    print("Accuracy of spatial relationship classifier on val set:", spatial_pipeline.score(np.load('val_spatial_features.npy'), y_val))
    joblib.dump(spatial_pipeline, 'spatial_pipeline.joblib')
    # Clear up memory
    del spatial_pipeline

    facs_pipeline = facial_unit_model(np.load('train_facs_features.npy'), y_train)
    probabilities_val["facs"] = facs_pipeline.predict_proba(np.load('val_facs_features.npy'))
    probabilities_test["facs"] = facs_pipeline.predict_proba(np.load('test_facs_features.npy'))
    # Log individual accuracy
    print("Accuracy of facial unit classifier on val set:", facs_pipeline.score(np.load('val_facs_features.npy'), y_val))
    joblib.dump(facs_pipeline, 'facs_pipeline.joblib')
    del facs_pipeline

    pdm_pipeline = pdm_model(np.load('train_pdm_features.npy'), y_train)
    probabilities_val["pdm"] = pdm_pipeline.predict_proba(np.load('val_pdm_features.npy'))
    probabilities_test["pdm"] = pdm_pipeline.predict_proba(np.load('test_pdm_features.npy'))
    # Log
    print("Accuracy of pdm classifier on val set:", pdm_pipeline.score(np.load('val_pdm_features.npy'), y_val))
    joblib.dump(pdm_pipeline, 'pdm_pipeline.joblib')
    del pdm_pipeline

    hog_pipeline = hog_model(np.load('train_hog_features.npy'), y_train)
    probabilities_val["hog"] = hog_pipeline.predict_proba(np.load('val_hog_features.npy'))
    probabilities_test["hog"] = hog_pipeline.predict_proba(np.load('test_hog_features.npy'))
    # Log
    print("Accuracy of hog classifier on val set:", hog_pipeline.score(np.load('val_hog_features.npy'), y_val))
    joblib.dump(hog_pipeline, 'hog_pipeline.joblib')
    del hog_pipeline


    logger.info("Starting Stacking...")

    stacking_pipe = evaluate_stacking(probabilities_val, y_val)
    joblib.dump(stacking_pipe, 'stacking_pipeline.joblib')

    def evaluate_test(stacking_pipe, y_test):
        logger.info("Evaluating Test Set...")
        X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

        stacking_accuracy = stacking_pipe.score(X_test_stack, y_test)

        logger.info(f"Accuracy of stacking classifier (Test Set): {stacking_accuracy}")

    evaluate_test(stacking_pipe, y_test)




