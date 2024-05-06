import os

import numpy as np

from src.algorithms.standardize_3d_landmarks import standardize_3d_landmarks


class DataLoader:
    def __init__(self, annotations_dir, features_dir):
        self.annotations_dir = annotations_dir
        self.features_dir = features_dir
        self._ids = self.filter_existing_features(self.get_ids())

        # Limit ids to 2500 (For Testing)
        # self._ids = self._ids[:2500]

        self.emotions = np.array(self.load_annotations()).astype(int)

        self._landmarks = []
        self._facs_intensity = []
        self._facs_presence = []
        #self._landmark_distances = []
        self._rigid_face_shape = []
        self._nonrigid_face_shape = []
        self._landmarks_3d = []
        self._hog = []
        self._deepface = []
        self._facenet = []


        self.features = {
            "landmarks": self._landmarks,
            "facs_intensity": self._facs_intensity,
            "facs_presence": self._facs_presence,
            #"landmark_distances": self._landmark_distances,
            "rigid_face_shape": self._rigid_face_shape,
            "nonrigid_face_shape": self._nonrigid_face_shape,
            "landmarks_3d": self._landmarks_3d,
            "hog": self._hog,
            "deepface": self._deepface,
            "facenet": self._facenet
        }

        self.load_features()

    # Loads y labels
    def load_annotations(self):
        emotions = []
        for sample in self._ids:
            emotions.append(np.load(f"{self.annotations_dir}/annotations/{sample}_exp.npy"))
        return emotions

    def get_ids(self):
        """
        Find all file ids in the features directory
        """
        ids = []
        for file in os.listdir(self.features_dir + "/features"):
            if file.endswith(".npy"):
                file_id = file.split("_")[0]
                if file_id not in ids:
                    ids.append(file_id)
        return ids


    def filter_existing_features(self, features):
        """
        Creates a list of ids, for which we have all feature types
        """
        ids = []
        for file_id in features:
            landmarks = f"{self.features_dir}/features/{file_id}_landmarks.npy"
            facs_intensity = f"{self.features_dir}/features/{file_id}_facs_intensity.npy"
            facs_presence = f"{self.features_dir}/features/{file_id}_facs_presence.npy"
            #landmark_distances = f"{self.features_dir}/features/{file_id}_landmark_distances.npy"
            rigid_face_shape = f"{self.features_dir}/features/{file_id}_rigid_face_shape.npy"
            nonrigid_face_shape = f"{self.features_dir}/features/{file_id}_nonrigid_face_shape.npy"
            landmarks_3d = f"{self.features_dir}/features/{file_id}_landmarks_3d.npy"
            hog = f"{self.features_dir}/features/{file_id}_hog.npy"
            if os.path.exists(landmarks) and os.path.exists(facs_intensity) and os.path.exists(facs_presence) and os.path.exists(rigid_face_shape) and os.path.exists(nonrigid_face_shape) and os.path.exists(landmarks_3d) and os.path.exists(hog):
                ids.append(file_id)
        return ids


    def check_features(self):
        # Make sure that for each image, we have all feature types
        all_ids = {}
        for file in os.listdir(self.features_dir + "/features"):
            if file.endswith(".npy"):
                file_id = file.split("_")[0]
                if file_id not in all_ids:
                    all_ids[file_id] = 0
                all_ids[file_id] += 1

        for file_id in all_ids:
            if all_ids[file_id] != list(all_ids.values())[0]:
                raise ValueError(f"Missing features for {file_id}")
        return all_ids

    # Loads X features
    def load_features(self):
        # First create list of all file ids, for which we have all feature types
        for file_id in self._ids:
            landmarks = np.load(f"{self.features_dir}/features/{file_id}_landmarks.npy")
            facs_intensity = np.load(f"{self.features_dir}/features/{file_id}_facs_intensity.npy")
            facs_presence = np.load(f"{self.features_dir}/features/{file_id}_facs_presence.npy")
            #landmark_distances = np.load(f"{self.features_dir}/features/{file_id}_landmark_distances.npy")
            rigid_face_shape = np.load(f"{self.features_dir}/features/{file_id}_rigid_face_shape.npy")
            nonrigid_face_shape = np.load(f"{self.features_dir}/features/{file_id}_nonrigid_face_shape.npy")
            hog = np.load(f"{self.features_dir}/features/{file_id}_hog.npy")[0]

            # Load and standardize 3d landmarks
            landmarks_3d = np.load(f"{self.features_dir}/features/{file_id}_landmarks_3d.npy")
            pose = np.load(f"{self.features_dir}/features/{file_id}_pose.npy")
            standardized_3d_landmarks = standardize_3d_landmarks(landmarks_3d, pose)

            deepface = np.load(f"{self.features_dir}/embeddings/{file_id}_DeepFace.npy")
            facenet = np.load(f"{self.features_dir}/embeddings/{file_id}_Facenet.npy")

            self._landmarks.append(landmarks)
            self._facs_intensity.append(facs_intensity)
            self._facs_presence.append(facs_presence)
            #self._landmark_distances.append(landmark_distances)
            self._rigid_face_shape.append(rigid_face_shape)
            self._nonrigid_face_shape.append(nonrigid_face_shape)
            self._landmarks_3d.append(standardized_3d_landmarks)
            self._hog.append(hog)
            self._facenet.append(facenet)
            self._deepface.append(deepface)





