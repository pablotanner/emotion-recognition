from src import get_important_features
from src.data_processing.embedding_loader import EmbeddingLoader
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM


embedding_loader = EmbeddingLoader("../../data")
X = embedding_loader.embeddings["SFace"]
y = embedding_loader.emotions

important_embedding_names = get_important_features(path='../../feature_scores_sface.npy', threshold=800)

new_X = []

for i in range(len(X)):
    data = []
    for name in important_embedding_names:
        index = int(name.split("_")[-1])
        data.append(X[i][index])
    new_X.append(data)


data_splitter = DataSplitter(new_X, y, test_size=0.2)
X_train, X_test, y_train, y_test = data_splitter.split_data()

svm = SVM(kernel='linear')
svm.train(X_train, y_train)
svm.evaluate(X_test, y_test)
