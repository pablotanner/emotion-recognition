# Visualize the shap values for the stacked model (hog, pdm, landmarks_3d, embedded, facs)
# Feature 0 is hog angry, feature 1 hog happy,....
# Feature 8 is pdm angry, feature 9 pdm happy,....

import shap
import numpy as np

shap_values = np.load('shap_values_test.npy')