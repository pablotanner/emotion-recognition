import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


from src import DataLoader, evaluate_results

data_loader = DataLoader("./data", "./data", exclude=['deepface', 'facenet', 'landmarks'])

y = data_loader.emotions

X_dict = data_loader.features
spatial_features = np.concatenate([X_dict['landmarks_3d'], X_dict['vggface']], axis=1)
facs_features = np.concatenate([X_dict['facs_intensity'], X_dict['facs_presence']], axis=1)
pdm_features = np.array(X_dict['nonrigid_face_shape'])
hog_features = np.array(X_dict['hog'])


X = np.concatenate([spatial_features, facs_features, pdm_features, hog_features], axis=1)

spatial_index = spatial_features.shape[1]
facs_index = facs_features.shape[1]
pdm_index = pdm_features.shape[1]
hog_index = hog_features.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Splitting data first

X_spatial_train, X_spatial_test = X_train[:, :spatial_index], X_test[:, :spatial_index]
X_facs_train, X_facs_test = X_train[:, spatial_index:spatial_index + facs_index], X_test[:, spatial_index:spatial_index + facs_index]
X_pdm_train, X_pdm_test = X_train[:, spatial_index + facs_index: spatial_index + facs_index + pdm_index], X_test[:, spatial_index + facs_index: spatial_index + facs_index + pdm_index]
X_hog_train, X_hog_test = X_train[:, spatial_index + facs_index + pdm_index:], X_test[:, spatial_index + facs_index + pdm_index:]



def spatial_relationship_model():
    # Linear scores worse individually, but better in stacking
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1, gamma='scale', kernel='linear', probability=True))
    ])

    svm_pipeline.fit(X_spatial_train, y_train)


    return svm_pipeline


def facial_unit_model():
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42)

    rf.fit(X_facs_train, y_train)

    return rf


def pdm_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    pipeline.fit(X_pdm_train, y_train)

    return pipeline


def hog_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # reduce dimensions
        ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='relu', random_state=42))

    ])

    pipeline.fit(X_hog_train, y_train)

    return pipeline

pipelines = {
    "spatial": spatial_relationship_model(),
    "facs": facial_unit_model(),
    "pdm": pdm_model(),
    "hog": hog_model()
}


# Probabilities for each model
probabilities = {
    "spatial": pipelines["spatial"].predict_proba(X_spatial_test),
    "facs": pipelines["facs"].predict_proba(X_facs_test),
    "pdm": pipelines["pdm"].predict_proba(X_pdm_test),
    "hog": pipelines["hog"].predict_proba(X_hog_test)
}

def evaluate_stacking(probabilities, pipelines):
    """
    Perform score fusion with stacking classifier
    """
    # Use probabilities as input to the stacking classifier
    X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(random_state=42))
    ])

    stacking_pipeline.fit(X_stack, y_test)

    stacking_accuracy = stacking_pipeline.score(X_stack, y_test)

    print("Accuracy of stacking classifier:", stacking_accuracy)

    # Individual accuracies
    print("Accuracy of individual models")
    test_sets = {
        "spatial": X_spatial_test,
        "facs": X_facs_test,
        "pdm": X_pdm_test,
        "hog": X_hog_test
    }

    # y_pred_spatial = pipelines["spatial"].predict(X_spatial_test)
    # evaluate_results(y_test, y_pred_spatial)

    for model in probabilities:
        accuracy = pipelines[model].score(test_sets[model], y_test)
        print(f"{model} accuracy: {accuracy}")


evaluate_stacking(probabilities, pipelines)