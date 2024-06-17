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

# Load the feature values
def violin_plot():
    # Show violin plot of SHAP values
    plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, show=False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    #plt.savefig('facs_violin_plot.png')
    plt.show()

violin_plot()