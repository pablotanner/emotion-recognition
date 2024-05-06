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


from src import DataLoader

data_loader = DataLoader("./data", "./data")

y = data_loader.emotions

X_dict = data_loader.features

spatial_features = np.concatenate([X_dict['landmarks'], X_dict['landmarks_3d']], axis=1)
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


spatial_pipeline = spatial_relationship_model()
facs_pipeline = facial_unit_model()
pdm_pipeline = pdm_model()
hog_pipeline = hog_model()

# Get Probabilities
spatial_probabilities = spatial_pipeline.predict_proba(X_spatial_test)
facs_probabilities = facs_pipeline.predict_proba(X_facs_test)
pdm_probabilities = pdm_pipeline.predict_proba(X_pdm_test)
hog_probabilities = hog_pipeline.predict_proba(X_hog_test)

def evaluate_stacking():
    """
    Perform score fusion with stacking classifier
    """
    # Use probabilities as input to the stacking classifier
    X_stack = np.column_stack((spatial_probabilities, facs_probabilities, pdm_probabilities, hog_probabilities))

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    stacking_pipeline.fit(X_stack, y_test)

    stacking_accuracy = stacking_pipeline.score(X_stack, y_test)

    print("Accuracy of stacking classifier:", stacking_accuracy)

    # Individual accuracies
    print("Accuracy of individual models")
    print("Spatial Relationship Model:", spatial_pipeline.score(X_spatial_test, y_test))
    print("Facial Unit Model:", facs_pipeline.score(X_facs_test, y_test))
    print("PDM Model:", pdm_pipeline.score(X_pdm_test, y_test))
    print("HOG Model:", hog_pipeline.score(X_hog_test, y_test))


def evaluate_majority_vote():
    """
    Perform score fusion with majority vote
    """

    # Convert probabilities to class predictions
    spatial_predictions = np.argmax(spatial_probabilities, axis=1)
    facs_predictions = np.argmax(facs_probabilities, axis=1)
    pdm_predictions = np.argmax(pdm_probabilities, axis=1)
    hog_predictions = np.argmax(hog_probabilities, axis=1)

    predictions = np.column_stack((spatial_predictions, facs_predictions, pdm_predictions, hog_predictions))

    # Perform majority voting
    majority_vote_predictions, _ = mode(predictions, axis=1)
    majority_vote_predictions = majority_vote_predictions.flatten()  # Flatten to get 1D array

    accuracy = accuracy_score(y_test, majority_vote_predictions)  # Ensure y_test is defined and correct
    print("Accuracy of majority voting classifier:", accuracy)

    # Individual accuracies
    print("Accuracy of individual models")
    print("Spatial Relationship Model:", accuracy_score(y_test, spatial_predictions))
    print("Facial Unit Model:", accuracy_score(y_test, facs_predictions))
    print("PDM Model:", accuracy_score(y_test, pdm_predictions))
    print("HOG Model:", accuracy_score(y_test, hog_predictions))

def evaluate_stacking_new():
    # Stack the probabilities to create a new training set for the meta-classifier
    X_stack_train = np.column_stack((spatial_probabilities, facs_probabilities, pdm_probabilities, hog_probabilities))

    # It's important that the stacking classifier itself is validated properly, possibly using cross-validation
    stacking_model = LogisticRegression()
    y_stack_pred = cross_val_predict(stacking_model, X_stack_train, y_test, cv=5)
    stacking_accuracy = accuracy_score(y_test, y_stack_pred)

    print("Accuracy of stacking classifier:", stacking_accuracy)


evaluate_stacking_new()