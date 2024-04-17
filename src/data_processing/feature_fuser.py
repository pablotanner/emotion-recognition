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


class Feature:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def preprocess(self):
        if self.name == 'landmarks':
            return preprocess_landmarks(self.data)
        if self.name == 'facs_intensity':
            return preprocess_facs_intensity(self.data)
        if self.name == 'facs_presence':
            return preprocess_facs_presence(self.data)
        if self.name == 'rigid_face_shape':
            return preprocess_rigid_face_shape(self.data)
        if self.name == 'nonrigid_face_shape':
            return preprocess_nonrigid_face_shape(self.data)
        if self.name == 'landmark_distances':
            return preprocess_landmark_distances(self.data)

    def __str__(self):
        return f"{self.name}: {self.data}"


class FusionStrategy:
    def preprocess_features(self, features):
        """ Process features as per the strategy. """
        raise NotImplementedError

    def postprocess_features(self, features):
        """ Scale or transform features after initial processing. """
        raise NotImplementedError


class KernelTransformerStrategy(FusionStrategy):
    def preprocess_features(self, features):
        return features  # Kernel transformation might be considered a form of postprocessing

    def postprocess_features(self, features):
        # Applying an RBF kernel as an example
        from sklearn.metrics.pairwise import rbf_kernel
        return rbf_kernel(features, gamma=0.1)

class StandardScalerStrategy(FusionStrategy):
    def preprocess_features(self, features):
        return features

    def postprocess_features(self, features):
        scaler = StandardScaler()
        return scaler.fit_transform(features)

class NoPostProcessing(FusionStrategy):
    def preprocess_features(self, features):
        return features

    def postprocess_features(self, features):
        # Return features as-is after initial preprocessing
        return features


class CompositeFusionStrategy(FusionStrategy):
    def __init__(self, strategies):
        self.strategies = strategies

    def preprocess_features(self, features):
        # Composite might not alter preprocessing, depends on use case
        return features

    def postprocess_features(self, features):
        for strategy in self.strategies:
            features = strategy.postprocess_features(features)
        return features




class FeatureFuser:
    def __init__(self, features_dict, include=None, fusion_strategy=None):
        self.features = features_dict
        self.include = include if include is not None else []
        self.fusion_strategy = fusion_strategy if fusion_strategy else NoPostProcessing()


    def fuse_features(self):
        processed_features = []


        for feature_name, feature_data in self.features.items():
            if feature_name not in self.include:
                continue

            feature = Feature(feature_name, feature_data)
            processed_features.append(feature.preprocess())


        fused_features = np.concatenate(processed_features, axis=1)

        fused_features = self.fusion_strategy.postprocess_features(fused_features)

        """
        if not use_scaler and not use_pca:
            return fused_features

        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(fused_features)

        if not use_pca:
            return standardized_features

        #dimensionality reduction with PCA
        pca = PCA(n_components=0.95, svd_solver='full')
        reduced_features = pca.fit_transform(standardized_features)
        """

        return fused_features

