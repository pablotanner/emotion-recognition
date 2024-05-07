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

#svc = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
#rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='relu', random_state=42)

#svc.__setattr__("__name__", "SVM")
#rf.__setattr__("__name__", "RandomForest")
mlp.__setattr__("__name__", "MLP")

# Train and Evaluate MLP using Stratified K Fold
for model in models:
    X = np.array(X_dict[model])
    print(f"Training and Evaluating {model}")
    perform_score_fusion(X, y, models=[mlp], n_splits=5, technique='average')
