import os
import numpy as np


def apply_transformation(landmarks, rx, ry, rz, tx, ty, tz):
    # Calculate pre-rotation center
    pre_center = np.mean(landmarks, axis=0)

    # Apply rotation
    R = rotation_matrix(rx, ry, rz)
    rotated_landmarks = np.dot(landmarks, R.T)

    # Calculate post-rotation center and adjust translation
    post_center = np.mean(rotated_landmarks, axis=0)
    translation_adjustment = pre_center - post_center

    # Invert translation
    tx, ty, tz = -tx, -ty, -tz

    final_translation = np.array([tx, ty, tz]) + translation_adjustment

    # Apply final translation
    transformed_landmarks = rotated_landmarks + final_translation

    return transformed_landmarks

def reshape_3d_landmarks(landmarks):
    """
    Transforms the extracted landmark data from [x0,x1,...y0,y1,...z0,z1,..] to [[x0,y0,z0],[x1,y1,z1],...]
    """
    X = landmarks[0:68]
    Y = landmarks[68:136]
    Z = landmarks[136:]
    return np.vstack((X, Y, Z)).T


def rotation_matrix(rx, ry, rz):
    """Create a 3x3 rotation matrix from rotation angles."""
    # Invert the rotation angles (So the direction is correct)
    rx, ry, rz = -rx, -ry, -rz
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    # Order of multiplication: first rotate around x, then y, then z
    return Rz @ Ry @ Rx


def standardize_3d_landmarks(landmarks, pose):
    """
    Standardize the landmarks by applying the pose transformation. First reshapes, then standardizes, then reshapes back.
    """
    reshaped_landmarks = reshape_3d_landmarks(landmarks)
    standardized_landmarks = apply_transformation(reshaped_landmarks, pose[3], pose[4], pose[5], pose[0], pose[1], pose[2])
    x, y, z = [], [], []
    for [x_coordinate, y_coordinate, z_coordinate] in standardized_landmarks:
        x.append(x_coordinate)
        y.append(y_coordinate)
        z.append(z_coordinate)
    return np.asarray(x + y + z)



# Data loader but specifically to how the data is stored on the rolf server
class RolfLoader:

    def __init__(self, main_annotations_dir, test_annotations_dir,  main_features_dir, test_features_dir, main_id_dir, excluded_features=None):
        self._main_id_dir = main_id_dir
        self._test_id_dir = []

        self._annotations_dir = {
            'train': main_annotations_dir,
            'val': main_annotations_dir,
            'test': test_annotations_dir
        }

        self._features_dir = {
            'train': main_features_dir,
            'val': main_features_dir,
            'test': test_features_dir
        }




        self.feature_types = [
            "landmarks",
            "facs_intensity",
            "facs_presence",
            "rigid_face_shape",
            "nonrigid_face_shape",
            "landmarks_3d",
            "hog",
            "landmarks_3d_unstandardized",
            #"deepface",
            #"facenet",
            #"vggface",
            #"openface",
            #"sface",
            #"facenet512",
            #"arcface",
        ]


        if excluded_features is None:
            self._excluded_features = []
        else:
            self._excluded_features = excluded_features


        self.features = {
            'train': {},
            'val': {},
            'test': {}
        }

        for feature_type in self.feature_types:
            if feature_type not in self._excluded_features:
                self.features['train'][feature_type] = []
                self.features['val'][feature_type] = []
                self.features['test'][feature_type] = []

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
        if self.id_is_complete(id, dataset_type):
            self.load_feature_files(id, dataset_type)
            self.load_annotations(id, dataset_type)


    def get_ids_and_data(self):
        """
        Gets ids for train and val data sets, then loads data
        """
        # Define datasets
        datasets = ['train', 'val', 'test']
        for dataset in datasets:
            if dataset == 'test':
                # For each '*_exp.npy' file in the test_annotations directory, take id and load data
                for file in os.listdir(self._annotations_dir[dataset]):
                    if file.endswith("_exp.npy"):
                        file_id = file.split('_')[0]
                        self.load_and_append_data(file_id, dataset)
            else:
                file_path = f'{self._main_id_dir}/{dataset}_ids.txt'
                with open(file_path, 'r') as file:
                    for id in file:
                        clean_id = id.strip()
                        self.load_and_append_data(clean_id, dataset)



    def id_is_complete(self, file_id, dataset_type):
        """
        Checks if for an id, all required feature types are present
        """
        features_path = f"{self._features_dir[dataset_type]}"


        if 'landmarks' not in self._excluded_features:
            landmarks = f"{features_path}/{file_id}_landmarks.npy"
            if not os.path.exists(landmarks):
                return False
        facs_intensity = f"{features_path}/{file_id}_facs_intensity.npy"
        if not os.path.exists(facs_intensity):
            return False
        facs_presence = f"{features_path}/{file_id}_facs_presence.npy"
        if not os.path.exists(facs_presence):
            return False
        rigid_face_shape = f"{features_path}/{file_id}_rigid_face_shape.npy"
        if not os.path.exists(rigid_face_shape):
            return False
        nonrigid_face_shape = f"{features_path}/{file_id}_nonrigid_face_shape.npy"
        if not os.path.exists(nonrigid_face_shape):
            return False
        if 'landmarks_3d' not in self._excluded_features:
            landmarks_3d = f"{features_path}/{file_id}_landmarks_3d.npy"
            if not os.path.exists(landmarks_3d):
                return False
        if 'hog' not in self._excluded_features:
            hog = f"{features_path}/{file_id}_hog.npy"
            if not os.path.exists(hog):
                return False
        return True

    def load_annotations(self, file_id, dataset_type):
        """
        Load the annotations for a file id
        """

        emotion = np.load(f"{self._annotations_dir[dataset_type]}/{file_id}_exp.npy")

        self.emotions[dataset_type].append(emotion)



    def load_feature_files(self, file_id, dataset_type):
        features_path = f"{self._features_dir[dataset_type]}/{file_id}"
        #embeddings_path = f"{self._embeddings_dir}/{file_id}"


        if 'landmarks' not in self._excluded_features:
            self.features[dataset_type]['landmarks'].append(np.load(f"{features_path}_landmarks.npy"))

        if 'hog' not in self._excluded_features:
            self.features[dataset_type]['hog'].append(np.load(f"{features_path}_hog.npy"))

        """

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

        
        """

        self.features[dataset_type]['facs_intensity'].append(np.load(f"{features_path}_facs_intensity.npy"))
        self.features[dataset_type]['facs_presence'].append(np.load(f"{features_path}_facs_presence.npy"))
        #data['landmark_distances'] = np.load(f"{features_path}_landmark_distances.npy")
        self.features[dataset_type]['rigid_face_shape'].append(np.load(f"{features_path}_rigid_face_shape.npy"))
        if 'nonrigid_face_shape' not in self._excluded_features:
            self.features[dataset_type]['nonrigid_face_shape'].append(np.load(f"{features_path}_nonrigid_face_shape.npy"))

        if 'landmarks_3d' not in self._excluded_features:
            pose = np.load(f"{features_path}_pose.npy")
            landmarks_3d = np.load(f"{features_path}_landmarks_3d.npy")
            self.features[dataset_type]['landmarks_3d'].append(standardize_3d_landmarks(landmarks_3d, pose))
            self.features[dataset_type]['landmarks_3d_unstandardized'].append(landmarks_3d)
