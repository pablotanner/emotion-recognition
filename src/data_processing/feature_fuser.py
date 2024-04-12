import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_landmarks(landmarks):
    # Normalize to range from 0 to 1
    max_value = max(np.array(landmarks).flatten())
    return landmarks / max_value

def preprocess_facs_intensity(facs_intensity):
    # Increase weight of the features
    return np.array(facs_intensity) * 10

def preprocess_facs_presence(facs_presence):
    return np.array(facs_presence) * 10

def preprocess_rigid_face_shape(rigid_face_shape):
    # Normalize to range from 0 to 1
    max_value = max(np.array(rigid_face_shape).flatten())

    return rigid_face_shape / max_value

def preprocess_nonrigid_face_shape(nonrigid_face_shape):
    # Normalize to range from 0 to 1
    max_value = max(np.array(nonrigid_face_shape).flatten())
    return nonrigid_face_shape / max_value

def preprocess_landmark_distances(landmark_distances):
    return landmark_distances



class FeatureFuser:
    def __init__(self, features_dict, include=None):
        if include is None:
            include = []
        self.features = features_dict
        self.include = include

    def fuse_features(self, use_scaler=False, use_pca=False):
        processed_features = []

        if 'landmarks' in self.include:
            processed_landmarks = preprocess_landmarks(self.features['landmarks'])
            processed_features.append(processed_landmarks)
        if 'facs_intensity' in self.include:
            processed_facs_intensity = preprocess_facs_intensity(self.features['facs_intensity'])
            processed_features.append(processed_facs_intensity)
        if 'facs_presence' in self.include:
            processed_facs_presence = preprocess_facs_presence(self.features['facs_presence'])
            processed_features.append(processed_facs_presence)
        if 'rigid_face_shape' in self.include:
            processed_rigid_face_shape = preprocess_rigid_face_shape(self.features['rigid_face_shape'])
            processed_features.append(processed_rigid_face_shape)
        if 'nonrigid_face_shape' in self.include:
            processed_nonrigid_face_shape = preprocess_nonrigid_face_shape(self.features['nonrigid_face_shape'])
            processed_features.append(processed_nonrigid_face_shape)
        if 'landmark_distances' in self.include:
            processed_landmark_distances = preprocess_landmark_distances(self.features['landmark_distances'])
            processed_features.append(processed_landmark_distances)


        fused_features = np.concatenate(processed_features, axis=1)

        if not use_scaler and not use_pca:
            return fused_features

        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(fused_features)

        if not use_pca:
            return standardized_features

        #dimensionality reduction with PCA
        pca = PCA(n_components=0.95, svd_solver='full')
        reduced_features = pca.fit_transform(standardized_features)

        return reduced_features

