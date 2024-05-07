import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.utils import to_categorical
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

data_loader = DataLoader("./data", "./data", exclude=[])

y = data_loader.emotions

X_dict = data_loader.features

def process_embeddings(embeddings):
    # Apply PCA then StandardSCaler
    pca = PCA(n_components=0.95)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    return embeddings

X_dict['sface'] = process_embeddings(X_dict['sface'])
X_dict['facenet'] = process_embeddings(X_dict['facenet'])
X_dict['arcface'] = process_embeddings(X_dict['arcface'])

spatial_features = np.concatenate([X_dict['landmarks_3d'], X_dict['landmarks']], axis=1)
embedded_features = np.concatenate([X_dict['sface'], X_dict['facenet'], X_dict['arcface']], axis=1)
facs_features = np.concatenate([X_dict['facs_intensity'], X_dict['facs_presence']], axis=1)
pdm_features = np.array(X_dict['nonrigid_face_shape'])
hog_features = np.array(X_dict['hog'])


X = np.concatenate([spatial_features, embedded_features, facs_features, pdm_features, hog_features], axis=1)

spatial_index = spatial_features.shape[1]
embedded_index = embedded_features.shape[1]
facs_index = facs_features.shape[1]
pdm_index = pdm_features.shape[1]
hog_index = hog_features.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Splitting data first

X_spatial_train, X_spatial_test = X_train[:, :spatial_index], X_test[:, :spatial_index]
X_embedded_train, X_embedded_test = X_train[:, spatial_index:spatial_index + embedded_index], X_test[:, spatial_index:spatial_index + embedded_index]
X_facs_train, X_facs_test = X_train[:, spatial_index + embedded_index:spatial_index + embedded_index + facs_index], X_test[:, spatial_index + embedded_index:spatial_index + embedded_index + facs_index]
X_pdm_train, X_pdm_test = X_train[:, spatial_index + embedded_index + facs_index:spatial_index + embedded_index + facs_index + pdm_index], X_test[:, spatial_index + embedded_index + facs_index:spatial_index + embedded_index + facs_index + pdm_index]
X_hog_train, X_hog_test = X_train[:, spatial_index + embedded_index + facs_index + pdm_index:], X_test[:, spatial_index + embedded_index + facs_index + pdm_index:]

def create_neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax') # 8 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Make so that if you do model.predict_proba, it calls model.predict
    model.predict_proba = model.predict
    return model

def spatial_relationship_model():
    # Linear scores worse individually, but better in stacking
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1, gamma='scale', kernel='linear', probability=True))
    ])

    pipeline.fit(X_spatial_train, y_train)


    return pipeline


def embedded_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # reduce dimensions
        ('svm', SVC(C=1, gamma='scale', kernel='rbf', probability=True))
    ])

    pipeline.fit(X_embedded_train, y_train)

    return pipeline



def facial_unit_model():
    """
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42)

    rf.fit(X_facs_train, y_train)

    return rf


    """

    nn = create_neural_network(X_facs_train.shape[1])
    y_train_categorical = to_categorical(y_train)
    nn.fit(X_facs_train, y_train_categorical, epochs=100, batch_size=32, verbose=0)

    return nn




def pdm_model():
    """

        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    pipeline.fit(X_pdm_train, y_train)

    return pipeline

        """

    nn = create_neural_network(X_pdm_train.shape[1])
    y_train_categorical = to_categorical(y_train)
    nn.fit(X_pdm_train, y_train_categorical, epochs=100, batch_size=32, verbose=0)

    return nn



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
    "embedded": embedded_model(),
    "facs": facial_unit_model(),
    "pdm": pdm_model(),
    "hog": hog_model()
}


# Probabilities for each model
probabilities = {
    "spatial": pipelines["spatial"].predict_proba(X_spatial_test),
    "embedded": pipelines["embedded"].predict_proba(X_embedded_test),
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

    evaluate_results(y_test, stacking_pipeline.predict(X_stack))

    # Individual accuracies
    print("Accuracy of individual models")
    test_sets = {
        "spatial": X_spatial_test,
        "embedded": X_embedded_test,
        "facs": X_facs_test,
        "pdm": X_pdm_test,
        "hog": X_hog_test
    }

    # y_pred_spatial = pipelines["spatial"].predict(X_spatial_test)
    # evaluate_results(y_test, y_pred_spatial)

    for model in probabilities:
        try:
            accuracy = pipelines[model].score(test_sets[model], y_test)
        except AttributeError:
            accuracy = accuracy_score(y_test, np.argmax(probabilities[model], axis=1))
        print(f"{model} accuracy: {accuracy}")


evaluate_stacking(probabilities, pipelines)