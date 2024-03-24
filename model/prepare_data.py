import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def prepare_data(fake_features=False, use_landmarks=True, use_facs_intensity=True, use_facs_presence=True, use_affect_net_lnd=False):
    num_samples = 5496
    X = []
    y = []
    for i in range(num_samples):
        # Check if the file exists
        try:
            emotion = np.load(f"../data/annotations/{i}_exp.npy")

            if use_affect_net_lnd:
                an_landmarks = np.load(f"../data/annotations/{i}_lnd.npy")
            else:
                an_landmarks = np.array([])
            if use_landmarks:
                landmarks = np.load(f"../data/features/{i}_landmarks.npy")
            else:
                landmarks = np.array([])

            if use_facs_intensity:
                facs_intensity = np.load(f"../data/features/{i}_facs_intensity.npy")
            else:
                facs_intensity = np.array([])

            if use_facs_presence:
                facs_presence = np.load(f"../data/features/{i}_facs_presence.npy")
            else:
                facs_presence = np.array([])

            # Combine the features
            features = np.concatenate((landmarks, an_landmarks, facs_intensity, facs_presence)).astype(float)

            if fake_features:
                # Generate fake features by randomizing
                # Assuming you know the shape of landmarks, facs_intensity, and facs_presence
                # For example, if landmarks are (68, 2), facs_intensity (12,), facs_presence (12,)
                landmarks_shape = (68 * 2,)  # Adjust according to actual shape, flattened
                facs_intensity_shape = (12,)  # Adjust according to actual shape
                facs_presence_shape = (12,)  # Adjust according to actual shape

                landmarks_fake = np.random.rand(*landmarks_shape)
                facs_intensity_fake = np.random.rand(*facs_intensity_shape)
                facs_presence_fake = np.random.rand(*facs_presence_shape)

                features = np.concatenate((landmarks_fake, facs_intensity_fake, facs_presence_fake)).astype(float)



        except FileNotFoundError:
            continue

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
