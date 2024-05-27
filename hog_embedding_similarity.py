
import joblib
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import CCA

if __name__ == '__main__':
    data_paths = {
        'hog': ['pca_train_hog_features.npy', 'pca_val_hog_features.npy', 'pca_test_hog_features.npy'],
        'embedded': ['train_embedded_features.npy', 'val_embedded_features.npy', 'test_embedded_features.npy'],
    }

    classifiers = {
        'hog': joblib.load('hog_pipeline.joblib'),
        'embedded': joblib.load('embedded_pipeline_mlp.joblib')
    }

    X_hog = np.load(data_paths['hog'][0])
    X_embeddings = np.load(data_paths['embedded'][0])
    cca = CCA(n_components=min(X_hog.shape[1], X_embeddings.shape[1]))
    X_hog_cca, X_embeddings_cca = cca.fit_transform(X_hog, X_embeddings)

    print("Canonical Correlations: ", cca.score(X_hog, X_embeddings))

    hog_similarity = np.sum(X_hog_cca * X_embeddings_cca, axis=1)
    hog_similarity = pd.Series(hog_similarity, name='hog_similarity')
    hog_similarity.to_csv('hog_similarity.csv', index=False)
    print(hog_similarity.head())
    print(hog_similarity.describe())
    print(hog_similarity.shape)








