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
from cuml.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import compute_class_weight

#from src.data_processing.rolf_loader import RolfLoader

parser = argparse.ArgumentParser(description='Model training and evaluation (GPU)')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)')
args = parser.parse_args()


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, num_classes, num_epochs=10, batch_size=32, learning_rate=0.001,
                 class_weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.model = self._build_model()

        if self.class_weights is not None:
            weight_tensor = torch.tensor(self.class_weights, dtype=torch.float32).cuda()
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
                        # save to file txt
                        filename='rolf_main_gpu.log',
                        )
    logger.info("Loading data...")
    #data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir, args.test_features_dir, args.main_id_dir)


    logger.info("Data loaded.")

    #feature_splits_dict, emotions_splits_dict = data_loader.get_data()

    # Dummy Data below
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

    logger.info("Data preprocessed.")


    def evaluate_stacking(probabilities, pipelines, X_val_spatial, X_val_facs, X_val_pdm, X_val_hog,
                          y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(random_state=42))
        ])

        stacking_pipeline.fit(X_stack, y_val)
        stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        log_reg_model = stacking_pipeline.named_steps['log_reg']
        coefficients = log_reg_model.coef_
        for idx, model_name in enumerate(probabilities.keys()):
            print(f"{coefficients[:, idx]}")

        print("Accuracy of stacking classifier (Val Set):", stacking_accuracy)
        # evaluate_results(y_val, stacking_pipeline.predict(X_stack))

        # Individual accuracies
        val_sets = {
            "spatial": X_val_spatial,
            "facs": X_val_facs,
            "pdm": X_val_pdm,
            "hog": X_val_hog
        }

        #evaluate_results(y_val, stacking_pipeline.predict(X_stack))

        for model in probabilities:
            try:
                accuracy = pipelines[model].score(val_sets[model], y_val)
            except AttributeError:
                accuracy = accuracy_score(y_val, np.argmax(probabilities[model], axis=1))
            print(f"{model} accuracy: {accuracy}")

        # Return the stacking pipeline
        return stacking_pipeline






    # From this we get X_train_spatial, X_val_spatial, X_test_spatial, X_train_embedded, etc.
    def get_feature_groups(features_dict):
        """
        :param features_dict: A dictionary containing the features of the different feature groups
        :return: A tuple containing the feature groups
        """

        spatial_features = np.concatenate([features_dict['landmarks_3d']], axis=1)
        # Concatenate selected embeddings
        facs_features = np.concatenate([features_dict['facs_intensity'], features_dict['facs_presence']], axis=1)
        pdm_features = np.array(features_dict['nonrigid_face_shape'])
        hog_features = np.array(features_dict['hog'])

        return spatial_features, facs_features, pdm_features, hog_features


    # Get the feature groups for the train, validation, and test sets
    X_train_spatial, X_train_facs, X_train_pdm, X_train_hog = get_feature_groups(feature_splits_dict['train'])
    X_val_spatial, X_val_facs, X_val_pdm, X_val_hog = get_feature_groups(feature_splits_dict['val'])
    X_test_spatial, X_test_facs, X_test_pdm, X_test_hog = get_feature_groups(feature_splits_dict['test'])

    # Get the emotions for the train, validation, and test sets
    y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']


    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)


    def spatial_relationship_model(X, y):
        # Linear scores worse individually, but better in stacking
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=0.1, probability=True, class_weight=class_weights)) #  kernel='linear', gamma='scale'
        ])

        pipeline.fit(X, y)

        return pipeline


    def facial_unit_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10))
        ])

        pipeline.fit(X, y)

        return pipeline


    def pdm_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(C=0.1, solver='qn', class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        return pipeline


    def hog_model(X, y):
        input_size = X.shape[1]
        hidden_size = 200
        num_classes = len(np.unique(y))  # Number of classes


        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),  # reduce dimensions
            ('mlp', PyTorchMLPClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,
                                         num_epochs=200, batch_size=32, learning_rate=0.001, class_weights=class_weights))
        ])

        pipeline.fit(X, y)

        return pipeline

    logger.info("Starting Fitting")


    pipelines = {
        "spatial": spatial_relationship_model(X_train_spatial, y_train),
        "facs": facial_unit_model(X_train_facs, y_train),
        "pdm": pdm_model(X_train_pdm, y_train),
        "hog": hog_model(X_train_hog, y_train)
    }

    # Probabilities for each model
    probabilities_val = {
        "spatial": pipelines["spatial"].predict_proba(X_val_spatial),
        "facs": pipelines["facs"].predict_proba(X_val_facs),
        "pdm": pipelines["pdm"].predict_proba(X_val_pdm),
        "hog": pipelines["hog"].predict_proba(X_val_hog)
    }

    logger.info("Starting Stacking")

    stacking_pipe = evaluate_stacking(probabilities_val, pipelines, X_val_spatial, X_val_facs, X_val_pdm, X_val_hog,
                                      y_val)

    # Finally, we can evaluate the stacking classifier on the test set
    probabilities_test = {
        "spatial": pipelines["spatial"].predict_proba(X_test_spatial),
        "facs": pipelines["facs"].predict_proba(X_test_facs),
        "pdm": pipelines["pdm"].predict_proba(X_test_pdm),
        "hog": pipelines["hog"].predict_proba(X_test_hog)
    }

    X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

    print("Accuracy of stacking classifier on test set:", stacking_pipe.score(X_test_stack, y_test))





