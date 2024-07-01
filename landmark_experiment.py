import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from src.model_training import SVC
from src import evaluate_results
from src.data_processing.rolf_loader import RolfLoader
from src.model_training.torch_neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description='Landmark Alignment Experiment')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
parser.add_argument('--data_output_dir', type=str, help='Path to the output directory', default='/local/scratch/ptanner/landmark_experiment')
parser.add_argument('--use_ss', type=bool, help='Whether to use standard scaler', default=False)
args = parser.parse_args()


if __name__ == '__main__':


    # If output directory files don't exist, save them
    if not os.path.exists(f'{args.data_output_dir}/train_landmarks_3d.npy') and not os.path.exists(f'{args.data_output_dir}/val_landmarks_3d.npy') and not os.path.exists(f'{args.data_output_dir}/test_landmarks_3d.npy') :
        print('Data not found, loading')
        data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                 args.test_features_dir, args.main_id_dir, excluded_features=['landmarks', 'hog'])
        feature_splits_dict, emotions_splits_dict = data_loader.get_data()
        y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict[
            'test']
        for split in ['train', 'val', 'test']:
            np.save(f'{args.data_output_dir}/{split}_landmarks_3d.npy', feature_splits_dict[split]['landmarks_3d'])
            np.save(f'{args.data_output_dir}/{split}_landmarks_3d_unstandardized.npy',
                    feature_splits_dict[split]['landmarks_3d_unstandardized'])
        del feature_splits_dict
    else:
        print('Found data, loading')
        y_train, y_val, y_test = np.load('y_train.npy'), np.load(
            'y_val.npy'), np.load('y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    standardized_results = {}
    unstandardized_results = {}


    def train_and_evaluate(X_train, y_train, X_test, y_test, is_standardized=True):
        print(20*'-')
        print("Standardized" if is_standardized else "Unstandardized")
        if args.use_ss:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        lr = LogisticRegression(C=10, class_weight='balanced')
        svc = SVC(C=10, class_weight='balanced', kernel='rbf')
        nn = NeuralNetwork(batch_size=128, num_epochs=10, class_weight=class_weights, input_dim=X_train.shape[1])

        models = {'SVC': svc, 'Logistic Regression': lr, 'NN': nn}

        for name, model in models.items():
            print(f"Training {name}")
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            print("Test set")
            evaluate_results(y_test, pred)
            if is_standardized:
                standardized_results[name] = model.score(X_test, y_test)
            else:
                unstandardized_results[name] = model.score(X_test, y_test)




    print("Training and evaluating on 3D landmarks")
    train_and_evaluate(np.load(f'{args.data_output_dir}/train_landmarks_3d.npy').astype(np.float32), y_train,
                       np.load(f'{args.data_output_dir}/test_landmarks_3d.npy').astype(np.float32), y_test,
                       is_standardized=True
                       )

    print("Training and evaluating on 3D landmarks unstandardized")
    train_and_evaluate(np.load(f'{args.data_output_dir}/train_landmarks_3d_unstandardized.npy').astype(np.float32), y_train,
                       np.load(f'{args.data_output_dir}/test_landmarks_3d_unstandardized.npy').astype(np.float32), y_test,
                       is_standardized=False
                       )












