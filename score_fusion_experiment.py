import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import evaluate_results
from src.data_processing.data_split_loader import DataSplitLoader

data_split_loader = DataSplitLoader("./data/annotations", "./data/features", "./data/embeddings", "./data/id_splits")

# Features and Emotions are split into train, validation, and test sets
feature_splits_dict, emotions_splits_dict = data_split_loader.get_data()


def process_embeddings(train_embeddings, val_embeddings, test_embeddings):
    # Apply PCA, then StandardScaler
    pca = PCA(n_components=0.95)
    pca.fit(train_embeddings)
    train_embeddings = pca.transform(train_embeddings)
    val_embeddings = pca.transform(val_embeddings)
    test_embeddings = pca.transform(test_embeddings)

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)
    test_embeddings = scaler.transform(test_embeddings)

    return train_embeddings, val_embeddings, test_embeddings


# Fit Embeddings and Transform
for key in ['sface', 'facenet', 'arcface']:
    train_embeddings, val_embeddings, test_embeddings = process_embeddings(feature_splits_dict['train'][key], feature_splits_dict['val'][key], feature_splits_dict['test'][key])
    feature_splits_dict['train'][key] = train_embeddings
    feature_splits_dict['val'][key] = val_embeddings
    feature_splits_dict['test'][key] = test_embeddings



# From this we get X_train_spatial, X_val_spatial, X_test_spatial, X_train_embedded, etc.
def get_feature_groups(features_dict):
    """
    :param features_dict: A dictionary containing the features of the different feature groups
    :return: A tuple containing the feature groups
    """

    spatial_features = np.concatenate([features_dict['landmarks_3d'], features_dict['landmarks']], axis=1)
    embedded_features = np.concatenate([features_dict['sface'], features_dict['facenet'], features_dict['arcface']], axis=1)
    facs_features = np.concatenate([features_dict['facs_intensity'], features_dict['facs_presence']], axis=1)
    pdm_features = np.array(features_dict['nonrigid_face_shape'])
    hog_features = np.array(features_dict['hog'])

    return spatial_features, embedded_features, facs_features, pdm_features, hog_features

# Get the feature groups for the train, validation, and test sets
X_train_spatial, X_train_embedded, X_train_facs, X_train_pdm, X_train_hog = get_feature_groups(feature_splits_dict['train'])
X_val_spatial, X_val_embedded, X_val_facs, X_val_pdm, X_val_hog = get_feature_groups(feature_splits_dict['val'])
X_test_spatial, X_test_embedded, X_test_facs, X_test_pdm, X_test_hog = get_feature_groups(feature_splits_dict['test'])

# Get the emotions for the train, validation, and test sets
y_train, y_val, y_test = emotions_splits_dict['train'], emotions_splits_dict['val'], emotions_splits_dict['test']


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

def spatial_relationship_model(X, y):
    # Linear scores worse individually, but better in stacking
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1, gamma='scale', kernel='linear', probability=True))
    ])

    pipeline.fit(X, y)


    return pipeline


def embedded_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # reduce dimensions
        ('svm', SVC(C=1, gamma='scale', kernel='rbf', probability=True))
        #('rf', RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42))
    ])

    pipeline.fit(X, y)

    return pipeline



def facial_unit_model(X, y):
    """

        nn = create_neural_network(X.shape[1])
    y_train_categorical = to_categorical(y)
    nn.fit(X, y_train_categorical, epochs=100, batch_size=32, verbose=0)

    return nn
    """
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42)

    rf.fit(X, y)

    return rf






def pdm_model(X, y):
    """
     nn = create_neural_network(X.shape[1])
    y_train_categorical = to_categorical(y)
    nn.fit(X, y_train_categorical, epochs=100, batch_size=32, verbose=0)

    return nn


    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

    pipeline.fit(X, y)

    return pipeline

def hog_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=0.95)),  # reduce dimensions
        ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='relu', random_state=42))

    ])

    pipeline.fit(X, y)

    return pipeline

pipelines = {
    "spatial": spatial_relationship_model(X_train_spatial, y_train),
    "embedded": embedded_model(X_train_embedded, y_train),
    "facs": facial_unit_model(X_train_facs, y_train),
    "pdm": pdm_model(X_train_pdm, y_train),
    "hog": hog_model(X_train_hog, y_train)
}


# Probabilities for each model
probabilities_val = {
    "spatial": pipelines["spatial"].predict_proba(X_val_spatial),
    "embedded": pipelines["embedded"].predict_proba(X_val_embedded),
    "facs": pipelines["facs"].predict_proba(X_val_facs),
    "pdm": pipelines["pdm"].predict_proba(X_val_pdm),
    "hog": pipelines["hog"].predict_proba(X_val_hog)
}

def evaluate_stacking(probabilities, pipelines, X_val_spatial, X_val_embedded, X_val_facs, X_val_pdm, X_val_hog, y_val):
    """
    Perform score fusion with stacking classifier
    """
    # Use probabilities as input to the stacking classifier
    X_stack = np.concatenate([probabilities[model] for model in probabilities], axis=1)

    stacking_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('log_reg', LogisticRegression(random_state=42))
    ])

    stacking_pipeline.fit(X_stack, y_val)
    stacking_accuracy = stacking_pipeline.score(X_stack, y_val)

    print("Accuracy of stacking classifier:", stacking_accuracy)

    evaluate_results(y_val, stacking_pipeline.predict(X_stack))

    # Individual accuracies
    print("Accuracy of individual models")
    val_sets = {
        "spatial": X_val_spatial,
        "embedded": X_val_embedded,
        "facs": X_val_facs,
        "pdm": X_val_pdm,
        "hog": X_val_hog
    }

    # y_pred_spatial = pipelines["spatial"].predict(X_spatial_test)
    # evaluate_results(y_test, y_pred_spatial)

    for model in probabilities:
        try:
            accuracy = pipelines[model].score(val_sets[model], y_val)
        except AttributeError:
            accuracy = accuracy_score(y_val, np.argmax(probabilities[model], axis=1))
        print(f"{model} accuracy: {accuracy}")


    # Return the stacking pipeline
    return stacking_pipeline

stacking_pipe = evaluate_stacking(probabilities_val, pipelines, X_val_spatial, X_val_embedded, X_val_facs, X_val_pdm, X_val_hog, y_val)


# Finally, we can evaluate the stacking classifier on the test set
probabilities_test = {
    "spatial": pipelines["spatial"].predict_proba(X_test_spatial),
    "embedded": pipelines["embedded"].predict_proba(X_test_embedded),
    "facs": pipelines["facs"].predict_proba(X_test_facs),
    "pdm": pipelines["pdm"].predict_proba(X_test_pdm),
    "hog": pipelines["hog"].predict_proba(X_test_hog)
}

X_test_stack = np.concatenate([probabilities_test[model] for model in probabilities_test], axis=1)

print("Accuracy of stacking classifier on test set:", stacking_pipe.score(X_test_stack, y_test))


