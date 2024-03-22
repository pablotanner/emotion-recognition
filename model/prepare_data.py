import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data():
    num_samples = 5496
    X = []
    y = []
    for i in range(num_samples):
        # Check if the file exists
        try:
            emotion = np.load(f"../data/annotations/{i}_exp.npy")
            landmarks = np.load(f"../data/features/{i}_landmarks.npy")
            facs_intensity = np.load(f"../data/features/{i}_facs_intensity.npy")
            facs_presence = np.load(f"../data/features/{i}_facs_presence.npy")

        except FileNotFoundError:
            continue
        # Combine the features
        features = np.concatenate((landmarks, facs_intensity, facs_presence)).astype(float)

        # Append the features and emotion to the lists
        X.append(features)
        y.append(int(emotion))


    return np.array(X), np.array(y)


def split_data(X, y, use_scaler=False):
    if use_scaler:
        # Standardizing the data
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X)
    else:
        X_final = X

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

