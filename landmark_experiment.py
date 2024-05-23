import argparse
import os

import numpy as np
from sklearn.utils import compute_class_weight
from cuml.svm import LinearSVC
from src import evaluate_results
from src.data_processing.rolf_loader import RolfLoader

parser = argparse.ArgumentParser(description='Landmark Alignment Experiment')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
parser.add_argument('--data_output_dir', type=str, help='Path to the output directory', default='/local/scratch/ptanner/landmark_experiment')
args = parser.parse_args()


if __name__ == '__main__':
    data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                             args.test_features_dir, args.main_id_dir, excluded_features=['landmarks', 'hog'])
    feature_splits_dict, emotions_splits_dict = data_loader.get_data()

    # If output directory files don't exist, save them
    if not os.path.exists(f'{args.data_output_dir}/train_landmarks_3d.npy') and not os.path.exists(f'{args.data_output_dir}/val_landmarks_3d.npy') and not os.path.exists(f'{args.data_output_dir}/test_landmarks_3d.npy') :
        for split in ['train', 'val', 'test']:
            np.save(f'{args.data_output_dir}/{split}_landmarks_3d.npy', feature_splits_dict[split]['landmarks_3d'])
            np.save(f'{args.data_output_dir}/{split}_landmarks_3d_unstandardized.npy',
                    feature_splits_dict[split]['landmarks_3d_unstandardized'])

    del feature_splits_dict

    y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict[
        'test']


    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, class_weights):
        svc = LinearSVC(C=1, probability=True, class_weight=class_weights)

        svc.fit(X_train, y_train)

        print("Validation set")
        evaluate_results(y_val, svc.predict(X_val))

        print("Test set")
        evaluate_results(y_test, svc.predict(X_test))



    print("Training and evaluating on 3D landmarks")
    train_and_evaluate(np.load(f'{args.data_output_dir}/train_landmarks_3d.npy'), y_train,
                       np.load(f'{args.data_output_dir}/val_landmarks_3d.npy'), y_val,
                       np.load(f'{args.data_output_dir}/test_landmarks_3d.npy'), y_test,
                       class_weights)

    print("Training and evaluating on 3D landmarks unstandardized")
    train_and_evaluate(np.load(f'{args.data_output_dir}/train_landmarks_3d_unstandardized.npy'), y_train,
                       np.load(f'{args.data_output_dir}/val_landmarks_3d_unstandardized.npy'), y_val,
                       np.load(f'{args.data_output_dir}/test_landmarks_3d_unstandardized.npy'), y_test,
                       class_weights)












