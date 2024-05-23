import argparse
import logging
import os

import numpy as np
import torch.optim as optim
#import cupy as cp
from cuml.svm import LinearSVC
from cuml.preprocessing import StandardScaler
#from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression as CUMLLogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from src.data_processing.rolf_loader import RolfLoader
import joblib
from datetime import datetime
from src.model_training.torch_mlp import PyTorchMLPClassifier
from src.model_training.torch_neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description='Model training and evaluation (GPU)')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
# Whether to use dummy data
parser.add_argument('--dummy', action='store_true', help='Use dummy data')
parser.add_argument('--use_existing',action='store_true', help='Use saved data/models')
parser.add_argument('--skip_hog', action='store_true', help='Skip HOG model training')
args = parser.parse_args()


if __name__ == "__main__":
    experiment_name = input("Enter experiment name: ")
    # Get ID for unique log file
    now = datetime.now()
    date = now.strftime("%m-%d")

    # If date directory doesn't exist, create it
    if not os.path.exists(f'logs/{date}'):
        os.makedirs(f'logs/{date}')

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'logs/{date}/{experiment_name}.log'),
                            logging.StreamHandler()
                        ])

    logger.info(f"Starting ROLF Training | Dummy: {args.dummy} | Use Existing: {args.use_existing}")
    logger.info("Loading data...")

    def save_features_to_disk(split_features_dict):
        """
        Save the features to disk
        """
        splits = list(split_features_dict.keys())

        for split in splits:
            np.save(f'{split}_spatial_features.npy', split_features_dict[split]['landmarks_3d'])
            del split_features_dict[split]['landmarks_3d']
            np.save(f'{split}_facs_features.npy', np.hstack([split_features_dict[split]['facs_intensity'], split_features_dict[split]['facs_presence']]))
            np.save(f'{split}_facs_intensity.npy', split_features_dict[split]['facs_intensity'])
            np.save(f'{split}_facs_presence.npy', split_features_dict[split]['facs_presence'])
            del split_features_dict[split]['facs_intensity']
            del split_features_dict[split]['facs_presence']
            np.save(f'{split}_pdm_features.npy', split_features_dict[split]['nonrigid_face_shape'])
            del split_features_dict[split]['nonrigid_face_shape']
            np.save(f'{split}_hog_features.npy', split_features_dict[split]['hog'])
            del split_features_dict[split]['hog']

            np.save(f'{split}_sface.npy', split_features_dict[split]['sface'])
            np.save(f'{split}_facenet.npy', split_features_dict[split]['facenet'])
            del split_features_dict[split]['sface']

            # Clear the dictionary to free up memory
            del split_features_dict[split]
            logger.info(f"Saved {split} features to disk")

    if not args.dummy:
        if not args.use_existing:
            data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir, args.test_features_dir, args.main_id_dir)
            feature_splits_dict, emotions_splits_dict = data_loader.get_data()
    else:
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

    logger.info("Data loaded.")

    def evaluate_stacking(probabilities, y_val):
        """
        Perform score fusion with stacking classifier
        """
        # Use probabilities as input to the stacking classifier
        X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

        stacking_pipeline = Pipeline([('log_reg', LogisticRegression(C=1, solver='liblinear', class_weight='balanced'))])

        stacking_pipeline.fit(X_stack, y_val)
        stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

        logger.info(f"Accuracy of stacking classifier (Validation Set): {stacking_accuracy}")

        balanced_accuracy = balanced_accuracy_score(y_val, stacking_pipeline.predict(X_stack))
        logger.info(f"Balanced Accuracy of stacking classifier (Validation Set): {balanced_accuracy}")

        # Return the stacking pipeline
        return stacking_pipeline


    # Get the emotions for the train, validation, and test sets
    if not args.use_existing:
        # Save features to disk and clear up from memory
        save_features_to_disk(feature_splits_dict)

        y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
        np.save('y_test.npy', y_test)
    else:
        y_train = np.load('y_train.npy')
        y_val = np.load('y_val.npy')
        y_test = np.load('y_test.npy')


    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    def spatial_relationship_model(X, y):
        # Linear scores worse individually, but better in stacking
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=1, probability=True, class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("Spatial Relationship Model Fitted")

        return pipeline


    def facial_unit_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp',
             PyTorchMLPClassifier(input_size=X.shape[1], hidden_size=300, num_classes=len(np.unique(y)), num_epochs=200,
                                  batch_size=32, learning_rate=0.001, class_weight=class_weights)
             )])

        pipeline.fit(X, y)

        logger.info("Facial Unit Model Fitted")

        return pipeline

    def nn_model(X, y):
        model = NeuralNetwork(input_dim=X.shape[1], class_weight=class_weights)
        model.compile(optim.Adam(model.parameters(), lr=0.001))
        model.fit(X, y)

        logger.info("NN Model Fitted")

        return model

    def rf_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("RF Model Fitted")

        return pipeline

    def log_reg_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', CUMLLogisticRegression(C=1, solver='qn', class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("Logistic Regression Model Fitted")

        return pipeline

    def embedded_model(X, y):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', PyTorchMLPClassifier(input_size=X.shape[1],
                                         hidden_size=300, num_classes=len(np.unique(y)),
                                         num_epochs=200, batch_size=32, learning_rate=0.001,
                                         class_weight=class_weights))])

        pipeline.fit(X, y)

        logger.info("Embedded Model Fitted")
        
        return pipeline


    def pdm_model(X, y):
        pipeline = Pipeline([
            #('scaler', StandardScaler()),
            ('mlp', PyTorchMLPClassifier(input_size=X.shape[1], hidden_size=300, num_classes=len(np.unique(y)), num_epochs=200, batch_size=32, learning_rate=0.01, class_weight=class_weights)
             )])

        pipeline.fit(X, y)

        logger.info("PDM Model Fitted")

        return pipeline

    def hog_model(X, y):
        pipeline = Pipeline([
            #('scaler', StandardScaler()),
            ('svm', LinearSVC(C=1, probability=True, class_weight=class_weights))
            #('mlp', PyTorchMLPClassifier(input_size=X.shape[1], hidden_size=300, num_classes=len(np.unique(y)), num_epochs=200, batch_size=32, learning_rate=0.001, class_weight=class_weights))
        ])

        pipeline.fit(X, y)

        logger.info("HOG Model Fitted")
        return pipeline

    logger.info("Starting Fitting...")


    probabilities_val = {}
    probabilities_test = {}

    # Train models, then save
    # if already trained, load from disk
    if os.path.exists('spatial_pipeline.joblib') and args.use_existing:
        spatial_pipeline = joblib.load('spatial_pipeline.joblib')
    else:
        spatial_pipeline = spatial_relationship_model(np.load('train_spatial_features.npy'), y_train)
        joblib.dump(spatial_pipeline, 'spatial_pipeline.joblib')
    probabilities_val["spatial"] = spatial_pipeline.predict_proba(np.load('val_spatial_features.npy'))
    probabilities_test["spatial"] = spatial_pipeline.predict_proba(np.load('test_spatial_features.npy'))
    # Log bal accs
    val_bal_acc = balanced_accuracy_score(y_val, spatial_pipeline.predict(np.load('val_spatial_features.npy')))
    #test_bal_acc = balanced_accuracy_score(y_test, spatial_pipeline.predict(np.load('test_spatial_features.npy')))
    logger.info(f"Balanced Accuracy of spatial relationship classifier on val set: {val_bal_acc}")
    #logger.info(f"Balanced Accuracy of spatial relationship classifier on test set: {test_bal_acc}")
    # Clear up memory
    del spatial_pipeline

    # Remove the combined facs pipeline, to see if it's better to separate them
    if os.path.exists('facs_pipeline.joblib') and args.use_existing:
        facs_pipeline = joblib.load('facs_pipeline.joblib')
    else:
        facs_pipeline = facial_unit_model(np.load('train_facs_features.npy'), y_train)
        joblib.dump(facs_pipeline, 'facs_pipeline.joblib')
    probabilities_val["facs"] = facs_pipeline.predict_proba(np.load('val_facs_features.npy'))
    probabilities_test["facs"] = facs_pipeline.predict_proba(np.load('test_facs_features.npy'))
    # Log bal acc
    val_bal_acc = balanced_accuracy_score(y_val, facs_pipeline.predict(np.load('val_facs_features.npy')))
    logger.info(f"Balanced Accuracy of facial unit classifier on val set: {val_bal_acc}")
    del facs_pipeline

    """
    facs_intensity_pipeline = facial_unit_model(np.load('train_facs_intensity.npy'), y_train)
    probabilities_val["facs_intensity"] = facs_intensity_pipeline.predict_proba(np.load('val_facs_intensity.npy'))
    probabilities_test["facs_intensity"] = facs_intensity_pipeline.predict_proba(np.load('test_facs_intensity.npy'))
    # Log balanced accuracy
    val_bal_acc = balanced_accuracy_score(y_val, facs_intensity_pipeline.predict(np.load('val_facs_intensity.npy')))
    #test_bal_acc = balanced_accuracy_score(y_test, facs_intensity_pipeline.predict(np.load('test_facs_intensity.npy')))
    logger.info(f"Balanced Accuracy of facs intensity classifier on val set: {val_bal_acc}")
    #logger.info(f"Balanced Accuracy of facs intensity classifier on test set: {test_bal_acc}")
    del facs_intensity_pipeline

    facs_presence_pipeline = rf_model(np.load('train_facs_presence.npy'), y_train)
    probabilities_val["facs_presence"] = facs_presence_pipeline.predict_proba(np.load('val_facs_presence.npy'))
    probabilities_test["facs_presence"] = facs_presence_pipeline.predict_proba(np.load('test_facs_presence.npy'))
    # Log bal accs
    val_bal_acc = balanced_accuracy_score(y_val, facs_presence_pipeline.predict(np.load('val_facs_presence.npy')))
    #test_bal_acc = balanced_accuracy_score(y_test, facs_presence_pipeline.predict(np.load('test_facs_presence.npy')))
    logger.info(f"Balanced Accuracy of facs presence classifier on val set: {val_bal_acc}")
    #logger.info(f"Balanced Accuracy of facs presence classifier on test set: {test_bal_acc}")
    del facs_presence_pipeline    
    
    """


    if os.path.exists('pdm_pipeline.joblib') and args.use_existing:
        pdm_pipeline = joblib.load('pdm_pipeline.joblib')
    else:
        pdm_pipeline = pdm_model(np.load('train_pdm_features.npy'), y_train)
        joblib.dump(pdm_pipeline, 'pdm_pipeline.joblib')
    probabilities_val["pdm"] = pdm_pipeline.predict_proba(np.load('val_pdm_features.npy'))
    probabilities_test["pdm"] = pdm_pipeline.predict_proba(np.load('test_pdm_features.npy'))
    # Log bal accs
    val_bal_acc = balanced_accuracy_score(y_val, pdm_pipeline.predict(np.load('val_pdm_features.npy')))
    #test_bal_acc = balanced_accuracy_score(y_test, pdm_pipeline.predict(np.load('test_pdm_features.npy')))
    logger.info(f"Balanced Accuracy of pdm classifier on val set: {val_bal_acc}")
    #logger.info(f"Balanced Accuracy of pdm classifier on test set: {test_bal_acc}")
    del pdm_pipeline

    if not args.skip_hog:
        if os.path.exists('hog_pipeline.joblib') and args.use_existing:
            hog_pipeline = joblib.load('hog_pipeline.joblib')
        else:
            if not os.path.exists('pca_train_hog_features.npy') or not os.path.exists('pca_val_hog_features.npy') or not os.path.exists('pca_test_hog_features.npy'):
                # Perform dimensionality reduction with PCA and save
                X_train_hog = np.load('train_hog_features.npy')
                logger.info("Fitting PCA for HOG training features...")
                pca = PCA(n_components=500)
                pca.fit(X_train_hog)
                # Save transformed features
                np.save('pca_train_hog_features.npy', pca.transform(X_train_hog))
                del X_train_hog
                # Transform val and test features
                X_val_hog = np.load('val_hog_features.npy')
                X_test_hog = np.load('test_hog_features.npy')
                logger.info("Transforming HOG val and test features...")
                np.save('pca_val_hog_features.npy', pca.transform(X_val_hog))
                np.save('pca_test_hog_features.npy', pca.transform(X_test_hog))
                del X_val_hog
                del X_test_hog
            logger.info("Fitting HOG model...")
            hog_pipeline = hog_model(np.load('pca_train_hog_features.npy'), y_train)
            joblib.dump(hog_pipeline, 'hog_pipeline.joblib')
        probabilities_val["hog"] = hog_pipeline.predict_proba(np.load('pca_val_hog_features.npy'))
        probabilities_test["hog"] = hog_pipeline.predict_proba(np.load('pca_test_hog_features.npy'))
        # Log bal accs
        val_bal_acc = balanced_accuracy_score(y_val, hog_pipeline.predict(np.load('pca_val_hog_features.npy')))
        #test_bal_acc = balanced_accuracy_score(y_test, hog_pipeline.predict(np.load('pca_test_hog_features.npy')))
        logger.info(f"Balanced Accuracy of HOG classifier on val set: {val_bal_acc}")
        #logger.info(f"Balanced Accuracy of HOG classifier on test set: {test_bal_acc}")
        del hog_pipeline

    if not os.path.exists('train_embedded_features.npy') or not os.path.exists('val_embedded_features.npy') or not os.path.exists('test_embedded_features.npy') or not args.use_existing:
        scaler = MinMaxScaler()
        X_train_fit_sface = scaler.fit_transform(np.load('train_sface.npy'))
        X_val_fit_sface = scaler.transform(np.load('val_sface.npy'))
        X_test_fit_sface = scaler.transform(np.load('test_sface.npy'))

        X_train_fit_facenet = scaler.fit_transform(np.load(f'{args.data_output_dir}/train_facenet.npy'))
        X_val_fit_facenet = scaler.transform(np.load(f'{args.data_output_dir}/val_facenet.npy'))
        X_test_fit_facenet = scaler.transform(np.load(f'{args.data_output_dir}/test_facenet.npy'))

        logger.info("Concatenating SFace and Facenet features...")
        X_train = np.concatenate([X_train_fit_sface, X_train_fit_facenet], axis=1)
        X_val = np.concatenate([X_val_fit_sface, X_val_fit_facenet], axis=1)
        X_test = np.concatenate([X_test_fit_sface, X_test_fit_facenet], axis=1)

        logger.info("Fitting PCA for embedded training features...")
        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

        np.save('train_embedded_features.npy', X_train)
        np.save('val_embedded_features.npy', X_val)
        np.save('test_embedded_features.npy', X_test)

        del X_train_fit_sface
        del X_val_fit_sface
        del X_test_fit_sface
        del X_train_fit_facenet
        del X_val_fit_facenet
        del X_test_fit_facenet
        del X_train
        del X_val
        del X_test
        del pca
        del scaler
    if os.path.exists('embedded_pipeline.joblib') and args.use_existing:
        embedded_pipeline = joblib.load('embedded_pipeline.joblib')
    else:
        embedded_pipeline = embedded_model(np.load('train_embedded_features.npy'), y_train)
        joblib.dump(embedded_pipeline, 'embedded_pipeline.joblib')
    probabilities_val["embedded"] = embedded_pipeline.predict_proba(np.load('val_embedded_features.npy'))
    probabilities_test["embedded"] = embedded_pipeline.predict_proba(np.load('test_embedded_features.npy'))
    # Log bal accs
    val_bal_acc = balanced_accuracy_score(y_val, embedded_pipeline.predict(np.load('val_embedded_features.npy')))
    #test_bal_acc = balanced_accuracy_score(y_test, embedded_pipeline.predict(np.load('test_embedded_features.npy')))
    logger.info(f"Balanced Accuracy of embedded classifier on val set: {val_bal_acc}")
    #logger.info(f"Balanced Accuracy of embedded classifier on test set: {test_bal_acc}")
    del embedded_pipeline

    logger.info("Starting Stacking...")
    stacking_pipe = evaluate_stacking(probabilities_val, y_val)

    logger.info("Classification Report (Val):")
    logger.info("\n" + classification_report(y_val, stacking_pipe.predict(np.concatenate([probabilities_val[model] for model in probabilities_val], axis=1))))

    def evaluate_test(stacking_pipe, y_test):
        logger.info("Evaluating Test Set...")
        X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

        stacking_accuracy = stacking_pipe.score(X_test_stack, y_test)

        logger.info(f"Accuracy of stacking classifier (Test Set): {stacking_accuracy}")
        logger.info(f"Balanced Accuracy of stacking classifier (Test Set): {balanced_accuracy_score(y_test, stacking_pipe.predict(X_test_stack))}")

        logger.info("Classification Report (Test):")
        logger.info("\n" + classification_report(y_test, stacking_pipe.predict(X_test_stack)))

    #evaluate_test(stacking_pipe, y_test)




