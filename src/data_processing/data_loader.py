"""
Not used in final thesis
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.algorithms.standardize_3d_landmarks import standardize_3d_landmarks


class DataLoader:
    def __init__(self, annotations_dir, features_dir, exclude=None):
        self.annotations_dir = annotations_dir
        self.features_dir = features_dir
        self.exclude = exclude
        if self.exclude is None:
            self.exclude = []
        self._ids = self.filter_existing_features(self.get_ids())
        # Limit ids to 2500 (For Testing)
        # self._ids = self._ids[:2500]

        self.emotions = np.array(self.load_annotations()).astype(int)

        self.features = {
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

        self.load_features_parallel()
        print("Loaded Data")

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

    def load_features_parallel(self):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.load_feature_files, file_id) for file_id in self._ids]
            for future in futures:
                result = future.result()
                for key in result:
                    self.features[key].append(result[key])


    def load_feature_files(self, file_id):
        data = {}
        features_path = f"{self.features_dir}/features/{file_id}"
        embeddings_path = f"{self.features_dir}/embeddings/{file_id}"

        if 'landmarks' not in self.exclude:
            data['landmarks'] = np.load(f"{features_path}_landmarks.npy", mmap_mode='r')

        if 'hog' not in self.exclude:
            data['hog'] = np.load(f"{features_path}_hog.npy", mmap_mode='r')[0]

        if 'deepface' not in self.exclude:
            data['deepface'] = np.load(f"{embeddings_path}_DeepFace.npy", mmap_mode='r')

        if 'facenet' not in self.exclude:
            data['facenet'] = np.load(f"{embeddings_path}_Facenet.npy", mmap_mode='r')

        if 'vggface' not in self.exclude:
            data['vggface'] = np.load(f"{embeddings_path}_VGG-Face.npy", mmap_mode='r')

        if 'openface' not in self.exclude:
            data['openface'] = np.load(f"{embeddings_path}_OpenFace.npy", mmap_mode='r')

        if 'sface' not in self.exclude:
            data['sface'] = np.load(f"{embeddings_path}_SFace.npy", mmap_mode='r')

        if 'facenet512' not in self.exclude:
            data['facenet512'] = np.load(f"{embeddings_path}_Facenet512.npy", mmap_mode='r')

        if 'arcface' not in self.exclude:
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



    # Loads X features
    def load_features(self):
        # First create list of all file ids, for which we have all feature types
        for file_id in self._ids:
            facs_intensity = np.load(f"{self.features_dir}/features/{file_id}_facs_intensity.npy")
            facs_presence = np.load(f"{self.features_dir}/features/{file_id}_facs_presence.npy")
            #landmark_distances = np.load(f"{self.features_dir}/features/{file_id}_landmark_distances.npy")
            rigid_face_shape = np.load(f"{self.features_dir}/features/{file_id}_rigid_face_shape.npy")
            nonrigid_face_shape = np.load(f"{self.features_dir}/features/{file_id}_nonrigid_face_shape.npy")

            # Load and standardize 3d landmarks
            landmarks_3d = np.load(f"{self.features_dir}/features/{file_id}_landmarks_3d.npy")
            pose = np.load(f"{self.features_dir}/features/{file_id}_pose.npy")
            standardized_3d_landmarks = standardize_3d_landmarks(landmarks_3d, pose)

            if 'landmarks' not in self.exclude:
                landmarks = np.load(f"{self.features_dir}/features/{file_id}_landmarks.npy")
                self.features['landmarks'].append(landmarks)

            if 'hog' not in self.exclude:
                hog = np.load(f"{self.features_dir}/features/{file_id}_hog.npy")[0]
                self.features['hog'].append(hog)

            if 'deepface' not in self.exclude:
                deepface = np.load(f"{self.features_dir}/embeddings/{file_id}_DeepFace.npy")
                self.features['deepface'].append(deepface)

            if 'facenet' not in self.exclude:
                facenet = np.load(f"{self.features_dir}/embeddings/{file_id}_Facenet.npy")
                self.features['facenet'].append(facenet)

            if 'vggface' not in self.exclude:
                vggface = np.load(f"{self.features_dir}/embeddings/{file_id}_VGG-Face.npy")
                self.features['vggface'].append(vggface)

            if 'openface' not in self.exclude:
                openface = np.load(f"{self.features_dir}/embeddings/{file_id}_OpenFace.npy")
                self.features['openface'].append(openface)

            self.features['facs_intensity'].append(facs_intensity)
            self.features['facs_presence'].append(facs_presence)
            #self.features['landmark_distances'].append(landmark_distances)
            self.features['rigid_face_shape'].append(rigid_face_shape)
            self.features['nonrigid_face_shape'].append(nonrigid_face_shape)
            self.features['landmarks_3d'].append(standardized_3d_landmarks)







