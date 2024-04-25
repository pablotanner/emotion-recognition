import numpy as np
from sklearn.svm import SVC

from src import EmbeddingLoader, DataSplitter, select_features

embedding_loader = EmbeddingLoader("../../data")

y = embedding_loader.emotions

# Went with S Face because lowest dimensionality and one of better accuracy scores
X = np.array(embedding_loader.embeddings["SFace"])

data_splitter = DataSplitter(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = data_splitter.split_data()

# Just use indexes for feature names
feature_names = [f"embedding_{i}" for i in range(len(X[0]))]
class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']


svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

feature_scores = select_features(svm, feature_names, X_train, y_train, X_test, y_test)