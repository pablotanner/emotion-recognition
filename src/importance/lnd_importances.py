import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Feature Names are coordinates, x_0,...x_67, y_0,...y_67, z_0,...z_67
feature_names = []
for coord in ['x', 'y', 'z']:
    for i in range(68):
        feature_names = feature_names + [f'{coord}_{i}']

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

# Load the SHAP values (n_samples, n_features, n_classes)
#shap_values = joblib.load('post_pca_shap_values.joblib')
shap_values = joblib.load('shap_values_NEW.joblib').values

shap_values = np.mean(np.abs(shap_values), axis=0)

# Create pandas dataframe with feature names as columns and class names as index
shap_df = pd.DataFrame(shap_values, columns=class_names, index=feature_names)

# First get sum of values for each coordinate
#values = {coord: np.array([shap_df[emotion][f'{coord}_{i}'] for i in range(68)]) for coord in ['x', 'y', 'z'] for emotion in class_names}

coordinate_scores = {emotion: {
    i: shap_df[emotion][f'x_{i}'] + shap_df[emotion][f'x_{i}'] + shap_df[emotion][f'z_{i}'] for i in range(68)
} for emotion in class_names}

# Fo each emotion, get 30 highest indexes
top_features = {emotion: {i: coordinate_scores[emotion][i] for i in sorted(coordinate_scores[emotion], key=coordinate_scores[emotion].get, reverse=True)[:68]} for emotion in class_names}

# Plot face image

def plot_face_image(top_dict):
    # Import image and draw points
    image_np = np.array(Image.open('face.png'))
    cords = np.load('../../data/features/0_landmarks.npy')
    #cords = np.load('test_lnd.npy')
    # Get rid of last third of coordinates (z)
    #cords = cords[:136]
    points = {i: [cords[i], cords[i+68]] for i in range(68)}

    # Resize image to 250x250
    image_np = Image.fromarray(image_np)
    image_np = image_np.resize((250, 250))
    image_np = np.array(image_np)

    #Shift points
    for point in points:
        x, y = points[point]
        x = (x + 5)
        y = (y - 35)
        points[point] = (x, y)

    # Stretch points
    for point in points:
        x, y = points[point]
        x = x * 1.05
        y = y * 1.2
        points[point] = (x, y)

    # Move points 0-6 to right
    for i in range(7):
        x, y = points[i]
        points[i] = (x + 5, y)

    # Move points 36-41 and 42-47 down
    for i in range(36, 42):
        x, y = points[i]
        points[i] = (x, y + 5)

    for i in range(42, 48):
        x, y = points[i]
        points[i] = (x, y + 5)

    # For point in top_dict.keys(), plot point, color based on value
    if top_dict is None:
        for point in points:
            x, y = points[point]
            plt.scatter(x, y, s=10, c='red')
    else:
        # Get colormap
        cmap = plt.cm.Reds
        values = np.array(list(top_dict.values()))
        normalized_values = (values - values.min()) / (values.max() - values.min())
        colors = cmap(normalized_values)
        color_list = {key: colors[i] for i, key in enumerate(top_dict.keys())}
        # Size based on value
        sizes = np.array(list(top_dict.values()))
        sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        sizes = sizes * 50

        for point in top_dict.keys():
            x, y = points[point]
            # Set color according to value usign colormap
            size = sizes[point]
            plt.scatter(x, y, s=size, color='blue') #color=color_list[point])

    # Hide Axis
    plt.axis('off')
    plt.imshow(image_np)
    plt.show()


plot_face_image(top_features['Happy'])
