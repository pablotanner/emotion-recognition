# Visualize the shap values for the stacked model (hog, pdm, landmarks_3d, embedded, facs)
# Feature 0 is hog angry, feature 1 hog happy,....
# Feature 8 is pdm angry, feature 9 pdm happy,....
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt

VARIANT = 0
emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

if VARIANT == 0:
    # Only correct predictions, only with the optimal models for stacking
    shap_values = joblib.load('SV_test_correct.joblib')
    # Aggregate for all classes, get from (n_samples, n_features, n_classes) to (n_samples, n_features)
    shap_values = np.mean(np.abs(shap_values.values), axis=2) # find betteR WAY, ABSOLUTE REMOVES DIRECTIONALITY
    # Not mean, it will be 0, sum

    #shap_values = shap_values[:, :, 0]



    models = ['HOG', 'PDM', 'Embeddings', 'FAUs']
else:
    # All models, all predictions
    shap_values = joblib.load('shap_test.joblib')[:, :, 1]
    models = ['HOG', 'PDM', '3D LND', 'Embeddings', 'FAUs']


real_feature_names = []
for model in models:
    for i in range(8):
        real_feature_names.append(model + ' ' + emotions[i])

#shap_values.feature_names = real_feature_names


def bar_plot():
    def get_mean_abs_shap_values(shap_values):
        return np.mean(np.abs(shap_values))

    # {feature: mean_abs_shap_value}
    features = {f: get_mean_abs_shap_values(shap_values[:, i]) for i, f in enumerate(real_feature_names)}

    # Dictionary of features and as value array of their shap values (one value per emotion)
    feature_shap_values = {m: [] for m in models}

    for model in models:
        for emotion in emotions:
            feature_shap_values[model].append(features[model + ' ' + emotion])

    # Create data plot
    df = pd.DataFrame(feature_shap_values, index=emotions).T

    colors = plt.get_cmap('tab20b')(np.linspace(0, 1, 8))
    ax = df.plot.barh(stacked=True, figsize=(14, 7), color=colors, align='center')
    plt.xlabel('mean( |SHAP Value| )', fontsize=18)
    plt.ylabel('Feature Type', fontsize=18)

    # add grid
    plt.grid(axis='x', color='white', linestyle='-')

    # Make sure grid is behind bars
    ax.set_axisbelow(True)

    # Create annotations
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if not width > 0.02:
            continue
        ax.text(x + width / 2,
                y + height / 2,
                '{:.2f}'.format(width),
                horizontalalignment='center',
                verticalalignment='center',
                color='white',
                fontsize=14)

    legend = plt.legend(loc='center',
                        frameon=False,
                        bbox_to_anchor=(0., 1.10, 1., .102),
                        mode='expand',
                        ncol=4,
                        borderaxespad=-.46,
                        prop={'size': 15, 'family': 'Calibri'})

    for text in legend.get_texts():
        plt.setp(text, color='black')

    title = plt.title('Feature Type Importances by Emotion', fontsize=20)

    # add space above
    plt.subplots_adjust(top=0.8)

    # add gray background
    plt.gca().set_facecolor('lightgray')

    plt.savefig('shap_values.pdf')
    plt.show()


bar_plot()
