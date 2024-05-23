import argparse
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight
from src import evaluate_results
from src.data_processing.rolf_loader import RolfLoader
from cuml.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.model_training.torch_mlp import PyTorchMLPClassifier

parser = argparse.ArgumentParser(description='Embedding EmoRec Experiment')
parser.add_argument('--main_annotations_dir', type=str, help='Path to /annotations folder (train and val)', default='/local/scratch/datasets/AffectNet/train_set/annotations')
parser.add_argument('--test_annotations_dir', type=str, help='Path to /annotations folder (test)', default='/local/scratch/datasets/AffectNet/val_set/annotations')
parser.add_argument('--main_features_dir', type=str, help='Path to /features folder (train and val)', default='/local/scratch/ptanner/features')
parser.add_argument('--test_features_dir', type=str, help='Path to /features folder (test)', default='/local/scratch/ptanner/test_features')
parser.add_argument('--main_id_dir', type=str, help='Path to the id files (e.g. train_ids.txt) (only for train and val)', default='/local/scratch/ptanner/')
parser.add_argument('--data_output_dir', type=str, help='Path to the output directory', default='/local/scratch/ptanner/embedding_experiment')
parser.add_argument('--use_existing', type=bool, help='Use existing data', default=False)
args = parser.parse_args()


if __name__ == '__main__':
    if not args.use_existing:
        data_loader = RolfLoader(args.main_annotations_dir, args.test_annotations_dir, args.main_features_dir,
                                 args.test_features_dir, args.main_id_dir, excluded_features=['landmarks', 'hog', 'landmarks_3d'])
        feature_splits_dict, emotions_splits_dict = data_loader.get_data()




        # If output directory files don't exist, save them
        for split in ['train', 'val', 'test']:
            np.save(f'{args.data_output_dir}/{split}_sface.npy', feature_splits_dict[split]['sface'])
            np.save(f'{args.data_output_dir}/{split}_facenet.npy', feature_splits_dict[split]['facenet'])

        del feature_splits_dict

        y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict[
            'test']
        np.save(f'{args.data_output_dir}/y_train.npy', y_train)
        np.save(f'{args.data_output_dir}/y_val.npy', y_val)
        np.save(f'{args.data_output_dir}/y_test.npy', y_test)

        del emotions_splits_dict

    y_train = np.load(f'{args.data_output_dir}/y_train.npy')
    y_val = np.load(f'{args.data_output_dir}/y_val.npy')
    y_test = np.load(f'{args.data_output_dir}/y_test.npy')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}


    # Just concatenate experiment
    def concatenated_experiment():
        print('Loading Data')
        X_train = np.concatenate([np.load(f'{args.data_output_dir}/train_sface.npy'), np.load(f'{args.data_output_dir}/train_facenet.npy')], axis=1)
        X_val = np.concatenate([np.load(f'{args.data_output_dir}/val_sface.npy'), np.load(f'{args.data_output_dir}/val_facenet.npy')], axis=1)
        X_test = np.concatenate([np.load(f'{args.data_output_dir}/test_sface.npy'), np.load(f'{args.data_output_dir}/test_facenet.npy')], axis=1)

        print('Preparing Pipeline')
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.99)),  # reduce dimensions,
            'mlp',
            PyTorchMLPClassifier(input_size=X_train.shape[1], hidden_size=300, num_classes=len(np.unique(y)), num_epochs=200,
                                 batch_size=32, learning_rate=0.001, class_weight=class_weights)
            ])

        print('Fitting Pipeline')
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, f"{args.data_output_dir}/embedded_pipeline.joblib")

        print("Validation set")
        evaluate_results(y_val, pipeline.predict(X_val))

        print("Test set")
        evaluate_results(y_test, pipeline.predict(X_test))

    concatenated_experiment()












