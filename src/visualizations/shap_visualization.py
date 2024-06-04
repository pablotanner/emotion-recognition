# Visualize the shap values for the stacked model (hog, pdm, landmarks_3d, embedded, facs)
# Feature 0 is hog angry, feature 1 hog happy,....
# Feature 8 is pdm angry, feature 9 pdm happy,....

import shap
import joblib
from matplotlib import pyplot as plt

emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

shap_values = joblib.load('shap_test.joblib')[:, :, 1]

models = ['HOG', 'PDM', '3D LND', 'Embeddings', 'FAUs']

real_feature_names = []
for model in models:
    for i in range(8):
        real_feature_names.append(model + ' ' + emotions[i])

shap_values.feature_names = real_feature_names


shap.plots.bar(shap_values, max_display=20, show=False)
# Add more space to left
plt.subplots_adjust(left=0.35)
plt.savefig('shap_bar_values.png')
plt.show()
plt.close()
