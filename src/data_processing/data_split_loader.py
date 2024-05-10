import concurrent
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.algorithms.standardize_3d_landmarks import standardize_3d_landmarks


class DataSplitLoader:
    def __init__(self, annotations_dir, features_dir, embeddings_dir, id_dir, excluded_features=None):
        self._annotations_dir = annotations_dir
        self._features_dir = features_dir
        self._embeddings_dir = embeddings_dir
        self._id_dir = id_dir
        if excluded_features is None:
            self._excluded_features = []
        else:
            self._excluded_features = excluded_features

        self.features = {
            'train': {
                "landmarks": [],
                "facs_intensity": [],
                "facs_presence": [],
                "rigid_face_shape": [],
                "nonrigid_face_shape": [],
                "landmarks_3d": [],
                "hog": [],
                "deepface": [],
                "facenet": [],
                "vggface": [],
                "openface": [],
                "sface": [],
                "facenet512": [],
                "arcface": [],
            },
            'val': {
                "landmarks": [],
                "facs_intensity": [],
                "facs_presence": [],
                "rigid_face_shape": [],
                "nonrigid_face_shape": [],
                "landmarks_3d": [],
                "hog": [],
                "deepface": [],
                "facenet": [],
                "vggface": [],
                "openface": [],
                "sface": [],
                "facenet512": [],
                "arcface": [],
            },
            'test': {
                "landmarks": [],
                "facs_intensity": [],
                "facs_presence": [],
                "rigid_face_shape": [],
                "nonrigid_face_shape": [],
                "landmarks_3d": [],
                "hog": [],
                "deepface": [],
                "facenet": [],
                "vggface": [],
                "openface": [],
                "sface": [],
                "facenet512": [],
                "arcface": [],
            }
        }

        self.emotions = {
            'train': [],
            'val': [],
            'test': []
        }

        self.get_ids_and_data()



    def get_data(self):
        return self.features, self.emotions


    def load_and_append_data(self, id, dataset_type):
        """
        Helper function to load and append data to the appropriate dataset
        """
        if self.id_is_complete(id):
            data = self.load_feature_files(id)
            annotation = self.load_annotations(id)
            if dataset_type == 'train':
                for key in data:
                    self.features['train'][key].append(data[key])
                self.emotions['train'].append(annotation)
            elif dataset_type == 'val':
                for key in data:
                    self.features['val'][key].append(data[key])
                self.emotions['val'].append(annotation)
            elif dataset_type == 'test':
                for key in data:
                    self.features['test'][key].append(data[key])
                self.emotions['test'].append(annotation)


    def get_ids_and_data(self):
        # Define datasets
        datasets = ['train', 'test', 'val']
        # Create a ThreadPoolExecutor to manage concurrency
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Handle each dataset files
            futures = []
            for dataset in datasets:
                file_path = f'{self._id_dir}/{dataset}_ids.txt'
                with open(file_path, 'r') as file:
                    for id in file:
                        clean_id = id.strip()
                        # Submit each file reading and processing to the executor
                        future = executor.submit(self.load_and_append_data, clean_id, dataset)
                        futures.append(future)
            # Wait for all futures to complete
            concurrent.futures.wait(futures)



    def id_is_complete(self, file_id):
        """
        Checks if for an id, all required feature types are present
        """
        if 'landmarks' not in self._excluded_features:
            landmarks = f"{self._features_dir}/{file_id}_landmarks.npy"
            if not os.path.exists(landmarks):
                return False
        facs_intensity = f"{self._features_dir}/{file_id}_facs_intensity.npy"
        if not os.path.exists(facs_intensity):
            return False
        facs_presence = f"{self._features_dir}/{file_id}_facs_presence.npy"
        if not os.path.exists(facs_presence):
            return False
        rigid_face_shape = f"{self._features_dir}/{file_id}_rigid_face_shape.npy"
        if not os.path.exists(rigid_face_shape):
            return False
        nonrigid_face_shape = f"{self._features_dir}/{file_id}_nonrigid_face_shape.npy"
        if not os.path.exists(nonrigid_face_shape):
            return False
        if 'landmarks_3d' not in self._excluded_features:
            landmarks_3d = f"{self._features_dir}/{file_id}_landmarks_3d.npy"
            if not os.path.exists(landmarks_3d):
                return False
        if 'hog' not in self._excluded_features:
            hog = f"{self._features_dir}/{file_id}_hog.npy"
            if not os.path.exists(hog):
                return False
        return True

    def load_annotations(self, file_id):
        """
        Load the annotations for a file id
        """
        annotations = np.load(f"{self._annotations_dir}/{file_id}_exp.npy")
        return annotations



    def load_feature_files(self, file_id):
        """ CAN BE OPTIMIZED, LOAD DIRECTLY INTO self.features"""
        data = {}
        features_path = f"{self._features_dir}/{file_id}"
        embeddings_path = f"{self._embeddings_dir}/{file_id}"

        if 'landmarks' not in self._excluded_features:
            data['landmarks'] = np.load(f"{features_path}_landmarks.npy", mmap_mode='r')

        if 'hog' not in self._excluded_features:
            data['hog'] = np.load(f"{features_path}_hog.npy", mmap_mode='r')[0]

        if 'deepface' not in self._excluded_features:
            data['deepface'] = np.load(f"{embeddings_path}_DeepFace.npy", mmap_mode='r')

        if 'facenet' not in self._excluded_features:
            data['facenet'] = np.load(f"{embeddings_path}_Facenet.npy", mmap_mode='r')

        if 'vggface' not in self._excluded_features:
            data['vggface'] = np.load(f"{embeddings_path}_VGG-Face.npy", mmap_mode='r')

        if 'openface' not in self._excluded_features:
            data['openface'] = np.load(f"{embeddings_path}_OpenFace.npy", mmap_mode='r')

        if 'sface' not in self._excluded_features:
            data['sface'] = np.load(f"{embeddings_path}_SFace.npy", mmap_mode='r')

        if 'facenet512' not in self._excluded_features:
            data['facenet512'] = np.load(f"{embeddings_path}_Facenet512.npy", mmap_mode='r')

        if 'arcface' not in self._excluded_features:
            data['arcface'] = np.load(f"{embeddings_path}_ArcFace.npy", mmap_mode='r')


        data['facs_intensity'] = np.load(f"{features_path}_facs_intensity.npy", mmap_mode='r')
        data['facs_presence'] = np.load(f"{features_path}_facs_presence.npy", mmap_mode='r')
        #data['landmark_distances'] = np.load(f"{features_path}_landmark_distances.npy")
        data['rigid_face_shape'] = np.load(f"{features_path}_rigid_face_shape.npy", mmap_mode='r')
        data['nonrigid_face_shape'] = np.load(f"{features_path}_nonrigid_face_shape.npy", mmap_mode='r')

        pose = np.load(f"{features_path}_pose.npy")
        landmarks_3d = np.load(f"{features_path}_landmarks_3d.npy", mmap_mode='r')
        data['landmarks_3d'] = standardize_3d_landmarks(landmarks_3d, pose)

        return data
