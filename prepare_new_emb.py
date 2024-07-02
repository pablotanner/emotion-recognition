import argparse

import numpy as np
import os

from sklearn.decomposition import IncrementalPCA as skIPCA
from cuml.decomposition import IncrementalPCA as cuMLIPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data_processing.rolf_loader import RolfLoader

parser = argparse.ArgumentParser()
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')

parser.add_argument('--cuml', action='store_true', help='Use cuml for PCA', default=False)
args = parser.parse_args()

if __name__ == '__main__':

    # Check if vggface and arcface train has been saved
    if os.path.exists('train_arcface.npy') and os.path.exists('train_vggface.npy'):
        print('Train embeddings already saved')
    else:
        data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                 args.test_features_dir, args.main_id_dir)
        feature_splits_dict, emotions_splits_dict = data_loader.get_data()
        splits = ['train', 'val', 'test']

        for split in splits:
            np.save(f'{split}_arcface.npy', feature_splits_dict[split]['arcface'])
            np.save(f'{split}_vggface.npy', feature_splits_dict[split]['vggface'])

        del data_loader, feature_splits_dict, emotions_splits_dict


    models = ['arcface','sface','facenet', 'vggface']

    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')

    for model in models:
        if not os.path.exists(f'embdata/train_{model}_ss.npy'):
            print(f'Scaling {model}')
            X_train = np.load(f'train_{model}.npy')
            X_val = np.load(f'val_{model}.npy')
            X_test = np.load(f'test_{model}.npy')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            np.save(f'embdata/train_{model}_ss.npy', X_train)
            np.save(f'embdata/val_{model}_ss.npy', X_val)
            np.save(f'embdata/test_{model}_ss.npy', X_test)
        else:
            print('Already SS scaled')
            X_train = np.load(f'embdata/train_{model}_ss.npy')
            X_val = np.load(f'embdata/val_{model}_ss.npy')
            X_test = np.load(f'embdata/test_{model}_ss.npy')

        if not os.path.exists(f'embdata/train_{model}_pca.npy'):
            print(f'PCA for {model}')
            # PCA
            if args.cuml:
                pca = cuMLIPCA(n_components=100)
            else:
                pca = skIPCA(n_components=100)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            X_test = pca.transform(X_test)
            np.save(f'embdata/train_{model}_pca.npy', X_train)
            np.save(f'embdata/val_{model}_pca.npy', X_val)
            np.save(f'embdata/test_{model}_pca.npy', X_test)
        else:
            print('Already PCA')

        if not os.path.exists(f'embdata/train_{model}_mm.npy'):
            print(f'MinMax for {model}')
            # MinMax
            minMaxScaler = MinMaxScaler(feature_range=(-5, 5))
            X_train = minMaxScaler.fit_transform(X_train)
            X_val = minMaxScaler.transform(X_val)
            X_test = minMaxScaler.transform(X_test)
            np.save(f'embdata/train_{model}_mm.npy', X_train)
            np.save(f'embdata/val_{model}_mm.npy', X_val)

        print(f'{model} done')


    # Concatenate all embeddings and save them
    X_train = np.concatenate([np.load(f'embdata/train_{model}_mm.npy') for model in models], axis=1)
    X_val = np.concatenate([np.load(f'embdata/val_{model}_mm.npy') for model in models], axis=1)
    X_test = np.concatenate([np.load(f'embdata/test_{model}_mm.npy') for model in models], axis=1)

    np.save('train_embeddings.npy', X_train)
    np.save('val_embeddings.npy', X_val)
    np.save('test_embeddings.npy', X_test)