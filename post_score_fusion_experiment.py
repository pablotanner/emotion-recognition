import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from src.data_processing.data_split_loader import DataSplitLoader
from sklearn.metrics import accuracy_score, classification_report

from src.feature_importance.feature_selection import select_features_adaboost

SELECTION_STRATEGY = 'filter'

data_split_loader = DataSplitLoader("./data/annotations", "./data/features", "./data/embeddings", "./data/id_splits")

features_to_use = ['nonrigid_face_shape', 'landmarks_3d', 'facs_intensity', 'hog']

features, emotions = data_split_loader.get_data()

# First apply PCA on HOG
pca = PCA(n_components=0.95)
features['train']['hog'] = pca.fit_transform(features['train']['hog'])
features['val']['hog'] = pca.transform(features['val']['hog'])
features['test']['hog'] = pca.transform(features['test']['hog'])

# Use MinMax Scaler for all
for feature in features_to_use:
    scaler = MinMaxScaler(feature_range=(-5,5))
    features['train'][feature] = scaler.fit_transform(features['train'][feature])
    features['val'][feature] = scaler.transform(features['val'][feature])
    features['test'][feature] = scaler.transform(features['test'][feature])

X_train = []
X_val = []
X_test = []

for feature in features_to_use:
    X_train.append(features['train'][feature])
    X_val.append(features['val'][feature])
    X_test.append(features['test'][feature])

X_train = np.concatenate(X_train, axis=1)
X_val = np.concatenate(X_val, axis=1)
X_test = np.concatenate(X_test, axis=1)

y_train = emotions['train']
y_val = emotions['val']
y_test = emotions['test']

if SELECTION_STRATEGY == 'filter':
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    selector = SelectKBest(f_classif, k=200)

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
elif SELECTION_STRATEGY == 'adaboost':
    X_train_selected, top_indices = select_features_adaboost(X_train, y_train, n_top_features=80)
    X_val_selected = X_val[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_val_selected = scaler.transform(X_val_selected)
    X_test_selected = scaler.transform(X_test_selected)


def create_neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax') # 8 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.__setattr__("__name__", "Sequential")
    model.predict_proba = model.predict
    return model


nn = create_neural_network(X_train_selected.shape[1])
# WITH HOG
svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, solver='sgd', learning_rate_init=0.001, activation='tanh', random_state=42)

# NO HOG
#svm = SVC(C=1, gamma='scale', kernel='rbf', probability=True, random_state=42)
#rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=10, random_state=42)
#mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, solver='sgd', learning_rate_init=0.001, activation='tanh', random_state=42)

nn.__setattr__("__name__", "Sequential")
svm.__setattr__("__name__", "SVM")
rf.__setattr__("__name__", "RandomForest")
mlp.__setattr__("__name__", "MLP")
models = [nn, svm, rf, mlp]


probabilities = {}
# Train the models
for model in models:
    if model.__name__ == "Sequential":
        y_train_categorical = to_categorical(y_train, num_classes=8)
        model.fit(X_train_selected, y_train_categorical, epochs=100, batch_size=32, verbose=0)
    else:
        model.fit(X_train_selected, y_train)
    # Get probabilities
    proba = model.predict_proba(X_val_selected)
    probabilities[model.__name__] = proba
    # Individual accuracy
    pred = np.argmax(proba, axis=1)
    print(f"{model.__name__} accuracy:", accuracy_score(y_val, pred))

# Get probabilities
probabilities = {
    "Sequential": nn.predict_proba(X_val_selected),
    "SVM": svm.predict_proba(X_val_selected),
    "RandomForest": rf.predict_proba(X_val_selected),
    "MLP": mlp.predict_proba(X_val_selected)
}

# Use logistic regression to perform score fusion
X_stack = np.concatenate(list(probabilities.values()), axis=1)

stacking_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression())
])

# Fit the logistic regression model on the stacked probabilities of the validation set
stacking_pipeline.fit(X_stack, y_val)

# Get accuracy on the validation set
y_pred = stacking_pipeline.predict(X_stack)
print("Accuracy (Val Set)", accuracy_score(y_val, y_pred))


def evaluate_test(X_test, y_test, models, stacking_pipeline):
    probabilities_test = {}
    for model in models:
        probabilities_test[model.__name__] = model.predict_proba(X_test)

    X_test_stack = np.concatenate(list(probabilities_test.values()), axis=1)

    y_pred = stacking_pipeline.predict(X_test_stack)

    print("Accuracy (Test Set)", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


evaluate_test(X_test_selected, y_test, models, stacking_pipeline)
