import argparse

parser = argparse.ArgumentParser(
    description='Training one classifier with all feature types')
parser.add_argument('--experiment-dir', type=str, help='Directory to checkpoint file',
                    default='/local/scratch/ptanner/single_classifier_experiments')
parser.add_argument('--classifier', type=str, help='Classifier to use', default='SVC')
args = parser.parse_args()


if __name__ == '__main__':
    features = ['hog', 'landmarks_3d', 'pdm', 'facs', 'embedded']

    for feature in features:
