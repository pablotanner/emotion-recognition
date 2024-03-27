import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(fake_features=False, use_scaler=True, use_landmarks=False, use_landmark_distances=False, use_facs_intensity=False, use_facs_presence=False, use_affect_net_lnd=False):
    num_samples = 5496 # There's not actually that many images in the dataset (2999), but the annotations are numbered up to 5496
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

            if use_landmark_distances:
                landmark_distances = np.load(f"../data/features/{i}_landmark_distances.npy")
            else:
                landmark_distances = np.array([])

            if use_facs_intensity:
                facs_intensity = np.load(f"../data/features/{i}_facs_intensity.npy")
            else:
                facs_intensity = np.array([])

            if use_facs_presence:
                facs_presence = np.load(f"../data/features/{i}_facs_presence.npy")
            else:
                facs_presence = np.array([])

            # Combine the features
            features = np.concatenate((facs_intensity, landmark_distances ,landmarks, an_landmarks, facs_presence)).astype(float)

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

    # Convert the lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Apply feature scaling
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)



    return X, y


def split_data(X, y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
