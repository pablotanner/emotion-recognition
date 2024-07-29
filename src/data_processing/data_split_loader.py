"""
Not used in final thesis
"""
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

        for key in self.emotions:
            self.emotions[key] = np.array(self.emotions[key]).astype(int)


    def get_data(self):
        return self.features, self.emotions


    def load_and_append_data(self, id, dataset_type):
        """
        Helper function to load and append data to the appropriate dataset
        """
        if self.id_is_complete(id):
            self.load_feature_files(id, dataset_type)
            self.load_annotations(id, dataset_type)


    def get_ids_and_data(self):
        # Define datasets
        datasets = ['train', 'test', 'val']
        for dataset in datasets:
            file_path = f'{self._id_dir}/{dataset}_ids.txt'
            with open(file_path, 'r') as file:
                for id in file:
                    clean_id = id.strip()
                    self.load_and_append_data(clean_id, dataset)




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

    def load_annotations(self, file_id, dataset_type):
        """
        Load the annotations for a file id
        """
        emotion = np.load(f"{self._annotations_dir}/{file_id}_exp.npy")
        self.emotions[dataset_type].append(emotion)



    def load_feature_files(self, file_id, dataset_type):
        """ CAN BE OPTIMIZED, LOAD DIRECTLY INTO self.features"""
        features_path = f"{self._features_dir}/{file_id}"
        embeddings_path = f"{self._embeddings_dir}/{file_id}"

        if 'landmarks' not in self._excluded_features:
            self.features[dataset_type]['landmarks'].append(np.load(f"{features_path}_landmarks.npy", mmap_mode='r'))

        if 'hog' not in self._excluded_features:
            self.features[dataset_type]['hog'].append(np.load(f"{features_path}_hog.npy", mmap_mode='r')[0])

        if 'deepface' not in self._excluded_features:
            self.features[dataset_type]['deepface'].append(np.load(f"{embeddings_path}_DeepFace.npy", mmap_mode='r'))

        if 'facenet' not in self._excluded_features:
            self.features[dataset_type]['facenet'].append(np.load(f"{embeddings_path}_Facenet.npy", mmap_mode='r'))

        if 'vggface' not in self._excluded_features:
            self.features[dataset_type]['vggface'].append(np.load(f"{embeddings_path}_VGG-Face.npy", mmap_mode='r'))

        if 'openface' not in self._excluded_features:
            self.features[dataset_type]['openface'].append(np.load(f"{embeddings_path}_OpenFace.npy", mmap_mode='r'))

        if 'sface' not in self._excluded_features:
            self.features[dataset_type]['sface'].append(np.load(f"{embeddings_path}_SFace.npy", mmap_mode='r'))

        if 'facenet512' not in self._excluded_features:
            self.features[dataset_type]['facenet512'].append(np.load(f"{embeddings_path}_Facenet512.npy", mmap_mode='r'))

        if 'arcface' not in self._excluded_features:
            self.features[dataset_type]['arcface'].append(np.load(f"{embeddings_path}_ArcFace.npy", mmap_mode='r'))


        self.features[dataset_type]['facs_intensity'].append(np.load(f"{features_path}_facs_intensity.npy", mmap_mode='r'))
        self.features[dataset_type]['facs_presence'].append(np.load(f"{features_path}_facs_presence.npy", mmap_mode='r'))
        #data['landmark_distances'] = np.load(f"{features_path}_landmark_distances.npy")
        self.features[dataset_type]['rigid_face_shape'].append(np.load(f"{features_path}_rigid_face_shape.npy", mmap_mode='r'))
        self.features[dataset_type]['nonrigid_face_shape'].append(np.load(f"{features_path}_nonrigid_face_shape.npy", mmap_mode='r'))

        pose = np.load(f"{features_path}_pose.npy")
        landmarks_3d = np.load(f"{features_path}_landmarks_3d.npy", mmap_mode='r')
        self.features[dataset_type]['landmarks_3d'].append(standardize_3d_landmarks(landmarks_3d, pose))
