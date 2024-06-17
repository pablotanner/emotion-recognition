import joblib
import shap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

feature_names = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
    'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c',
    'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c',
    'AU28_c', 'AU45_c'
]

# Load the SHAP values
shap_values = joblib.load('shap_facs.joblib')

shap_values.feature_names = feature_names

positive_shap_values_array = np.where(shap_values.values > 0, shap_values.values, 0)
mean_positive_shap_values = np.mean(positive_shap_values_array, axis=0)

#mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

#shap_df = pd.DataFrame(mean_abs_shap_values, columns=class_names, index=feature_names)

shap_df = pd.DataFrame(mean_positive_shap_values, columns=class_names, index=feature_names)

#shap_df = shap_df.sort_values(by='Happy', ascending=False)

# For each emotion, get the top 5 features
top_features = {emotion: shap_df[emotion].sort_values(ascending=False).head(3) for emotion in class_names}

def get_unique_faus(features_dict):
    unique = set()
    for k in features_dict.keys():
        for facs in list(top_features[k].keys()):
            unique.add(facs)

    return unique

