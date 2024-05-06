import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src import EmbeddingLoader, DataSplitter
from src.model_training.score_fusion import perform_score_fusion

embedding_loader = EmbeddingLoader("../../data")

y = embedding_loader.emotions

X_dict = embedding_loader.embeddings


models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "SFace"
]

svc = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=400, random_state=42)

svc.__setattr__("__name__", "SVM")
rf.__setattr__("__name__", "RandomForest")
mlp.__setattr__("__name__", "MLP")

classifiers = [
    svc, rf, mlp
]


for model in models[2:]:
    # Split and Scale data, then concatenate again
    X = np.array(X_dict[model])
    print(10*"=" + f"Model: {model}" + 10*"=")
    perform_score_fusion(X, y, models=classifiers, n_splits=5, technique='average')


