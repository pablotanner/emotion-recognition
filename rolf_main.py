import argparse
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.data_processing.rolf_loader import RolfLoader

parser = argparse.ArgumentParser(description='Model training and evaluation.')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)')
args = parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading data...")
    data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir, args.test_features_dir, args.main_id_dir)
    logger.info("Data loaded.")

    feature_splits_dict, emotions_splits_dict = data_loader.get_data()

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
            # "embedded": X_val_embedded,
            "facs": X_val_facs,
            "pdm": X_val_pdm,
            "hog": X_val_hog
        }

        # y_pred_spatial = pipelines["spatial"].predict(X_spatial_test)
        # evaluate_results(y_test, y_pred_spatial)

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


    def spatial_relationship_model(X, y):
        # Linear scores worse individually, but better in stacking
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(C=0.1, gamma='scale', kernel='linear', probability=True))
        ])

        pipeline.fit(X, y)

        return pipeline


    def facial_unit_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42))
        ])

        pipeline.fit(X, y)

        return pipeline


    def pdm_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(C=0.1, solver='liblinear', random_state=42))
        ])

        pipeline.fit(X, y)

        return pipeline


    def hog_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),  # reduce dimensions
            ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=200, solver='sgd', learning_rate_init=0.001,
                                  activation='relu', random_state=42))
        ])

        pipeline.fit(X, y)

        return pipeline


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





