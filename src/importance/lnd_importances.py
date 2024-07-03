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
shap_values = joblib.load('post_pca_shap_values.joblib')
#shap_values = joblib.load('shap_values_NEW.joblib').values

# Only positive
#shap_values = np.where(shap_values > 0, shap_values, 0)

shap_values = np.mean(np.abs(shap_values), axis=0)

# Create pandas dataframe with feature names as columns and class names as index
shap_df = pd.DataFrame(shap_values, columns=class_names, index=feature_names)

# First get sum of values for each coordinate
#values = {coord: np.array([shap_df[emotion][f'{coord}_{i}'] for i in range(68)]) for coord in ['x', 'y', 'z'] for emotion in class_names}

coordinate_scores = {emotion: {
    i: shap_df[emotion][f'x_{i}'] + shap_df[emotion][f'y_{i}'] + shap_df[emotion][f'z_{i}'] for i in range(68)
} for emotion in class_names}


# Fo each emotion, get 30 highest indexes
top_features = {emotion: {i: coordinate_scores[emotion][i] for i in sorted(coordinate_scores[emotion], key=coordinate_scores[emotion].get, reverse=True)[:68]} for emotion in class_names}

# Plot face image

def plot_face_image(emo):
    top_dict = top_features[emo]
    # Import image and draw points
    image_np = np.array(Image.open('face.png'))
    cords = np.load('../../data/features/0_landmarks.npy')

    # To convert from 3d lnd, load like this
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

    # Plot the image first
    plt.imshow(image_np)
    plt.axis('off')

    # For point in top_dict.keys(), plot point, color based on value
    if top_dict is None:
        for point in points:
            x, y = points[point]
            plt.scatter(x, y, s=10, c='red')
    else:
        # Get values from all points in the top_features dict, normalize them
        # GLOBAL NORMALIZATION
        #values = []
        #for emot in top_features.keys():
            #for point in top_features[emot].keys():
               # values.append(top_features[emot][point])
        #max_val = max(values)
        #min_val = min(values)
        #normalized_values = [(val - min_val) / (max_val - min_val) for val in values]


        # LOCAL NORMALIZATION
        normalized_values = []
        for point in top_dict.keys():
            normalized_values.append(top_dict[point])
        max_val = max(normalized_values)
        min_val = min(normalized_values)
        normalized_values = [(val - min_val) / (max_val - min_val) for val in normalized_values]


        for point in top_dict.keys():
            # Use normalized value to set size, use
            # get 'position' of point in all values
            size = normalized_values[list(top_dict.keys()).index(point)] * 50
            x, y = points[point]
            color = 'red'
            # If value is in top 5, color green
            if point in sorted(top_dict, key=top_dict.get, reverse=True)[:5]:
                color = 'lawngreen'

            plt.scatter(x, y, s=size, color=color)

    # Hide Axis
    plt.title(emo, fontsize=26)
    plt.savefig(f'{emo}_shap.png', bbox_inches='tight',transparent=True, pad_inches=0)
    plt.show()
    plt.close()

for emo in class_names:
    print(emo)
    plot_face_image(emo)
#import math
#for i in range(68):
    #print(f'{round(coordinate_scores["Neutral"][i], 4)}  {round(coordinate_scores["Happy"][i], 4)}  {round(coordinate_scores["Sad"][i], 4)}  {round(coordinate_scores["Surprise"][i], 4)}  {round(coordinate_scores["Fear"][i], 4)}  {round(coordinate_scores["Disgust"][i], 4)}  {round(coordinate_scores["Angry"][i], 4)}  {round(coordinate_scores["Contempt"][i], 4)}')