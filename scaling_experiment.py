import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.model_training import SVC
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.model_training.mlp_classifier import MLP as MLP
from src.model_training.sequentialnn_classifier import SequentialNN
from src.data_processing.rolf_loader import RolfLoader

"""
Code for preliminary experiments for "Normalization via StandardScaler" in Section 6.1.1
"""

parser = argparse.ArgumentParser(description='Scaling Experiment')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')

args = parser.parse_args()


if __name__ == '__main__':
    data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                             args.test_features_dir, args.main_id_dir, excluded_features=['landmarks', 'hog'])
    feature_splits_dict, emotions_splits_dict = data_loader.get_data()
    y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict[
        'test']

    lnd_train = np.array(feature_splits_dict['train']['landmarks_3d'])
    #lnd_val = feature_splits_dict['val']['landmarks_3d']
    lnd_test = np.array(feature_splits_dict['test']['landmarks_3d'])

    pdm_train = np.array(feature_splits_dict['train']['nonrigid_face_shape'])
    #pdm_val = feature_splits_dict['val']['nonrigid_face_shape']
    pdm_test = np.array(feature_splits_dict['test']['nonrigid_face_shape'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    unscaled_results = {}
    scaled_results = {}

    def scaled_experiment(X_train, y_train, X_test, y_test):

        lr = LogisticRegression(C=10, class_weight='balanced')
        lr.fit(X_train, y_train)
        unscaled_results['LogisticRegression'] = lr.score(X_test, y_test)

        del lr

        print(f"Logistic Regression: {unscaled_results['LogisticRegression']}")

        mlp = MLP(hidden_size=256, batch_size=64, class_weight=class_weights, learning_rate=0.01, num_epochs=20,
                  num_classes=8, input_size=X_test.shape[1])
        mlp.fit(X_train, y_train)
        unscaled_results['MLP'] = mlp.score(X_test, y_test)

        del mlp

        print(f"MLP: {unscaled_results['MLP']}")

        nn = SequentialNN(batch_size=64, num_epochs=20, class_weight=class_weights, input_dim=X_test.shape[1],
                          learning_rate=0.01, )
        nn.fit(X_train, y_train)
        unscaled_results['SequentialNN'] = nn.score(X_test, y_test)

        del nn

        print(f"SequentialNN: {unscaled_results['SequentialNN']}")

        svc = SVC(C=1, class_weight='balanced', kernel='rbf')
        svc.fit(X_train, y_train)
        unscaled_results['SVC'] = svc.score(X_test, y_test)

        del svc

        print(f"SVC: {unscaled_results['SVC']}")

        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
        rf.fit(X_train, y_train)
        unscaled_results['RandomForest'] = rf.score(X_test, y_test)

        del rf

        print(f"Random Forest: {unscaled_results['RandomForest']}")


    print("3D Landmarks NO SCALING")
    scaled_experiment(lnd_train, y_train, lnd_test, y_test)

    print("PDM NO SCALING")
    scaled_experiment(pdm_train, y_train, pdm_test, y_test)

    scaler = StandardScaler()
    lnd_train = scaler.fit_transform(lnd_train)
    lnd_test = scaler.transform(lnd_test)

    scaler = StandardScaler()
    pdm_train = scaler.fit_transform(pdm_train)
    pdm_test = scaler.transform(pdm_test)

    print("3D Landmarks SCALED")
    scaled_experiment(lnd_train, y_train, lnd_test, y_test)

    print("PDM SCALED")
    scaled_experiment(pdm_train, y_train, pdm_test, y_test)











