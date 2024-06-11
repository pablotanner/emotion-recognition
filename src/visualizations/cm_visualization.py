import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


key_map = {
    'facs': 'FAUs',
    'landmarks_3d': '3D Landmarks',
    'pdm': 'PDM',
    'embedded': 'Embeddings',
    'hog': 'HOG',
    'concat': 'Concatenated',
    'aggregated': 'Aggregated'
}

# Load normalized confusion matrices for each model/feature
#confusion_matrices = np.load('cm_matrices.npy', allow_pickle=True).item()
confusion_matrices = np.load('cm_matrices_not_normalized.npy', allow_pickle=True).item()
emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

# Get averages for all models except for FACS
average_confusion_matrices = {}
for model, matrix in confusion_matrices.items():
    average_confusion_matrices[model] = matrix

# Get the average confusion matrix
average_confusion_matrix = np.round(np.mean([matrix for matrix in average_confusion_matrices.values()], axis=0))

def plot_cm(confusion_matrix):
    # Pad and print
    print(np.around(confusion_matrix))

    # Visualize as Stacked Bar plot, classes on x-axis, y axis is the percentage of items correctly/incorrectly classified
    plt.figure(figsize=(10, 10))

    # One bar per Emotion, stacked for Correctly Classified, then also stacked for incorrectly classified as Main Class and Other Class
    correctly = np.diagonal(confusion_matrix)  # true positives
    # False Negatives
    incorrectly = np.sum(confusion_matrix, axis=1) - correctly
    # True negatives
    # true_negatives = np.sum(average_confusion_matrix) - np.sum(incorrectly)
    # False Positives
    incorrectly_other = np.sum(confusion_matrix, axis=0) - correctly

    # False Positives
    plt.bar(np.arange(8), incorrectly_other, bottom=correctly + incorrectly, label='False Positives', color='#E6E8E9')
    # True Positives
    plt.bar(np.arange(8), correctly, label='True Positives', color='#24AAE1')
    # False Negatives, Samples of this class that were wrongly classified as another class
    plt.bar(np.arange(8), incorrectly, bottom=correctly, label='False Negatives', color='#BE1E2C')

    # Add black border around TP and FN bars
    plt.bar(np.arange(8), correctly + incorrectly, color='none', edgecolor='black', linewidth=1,
            label='Total Samples (TP + FN)'
            )

    # Add Numbers (whole numbers)
    for i in range(8):
        plt.text(i, correctly[i] / 2, str(int(correctly[i])), ha='center', va='center', color='black')
        plt.text(i, correctly[i] + incorrectly[i] / 2, str(int(incorrectly[i])), ha='center', va='center',
                 color='black')
        plt.text(i, correctly[i] + incorrectly[i] + incorrectly_other[i] / 2, str(int(incorrectly_other[i])),
                 ha='center', va='center', color='black')

    # Grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.xticks(np.arange(8), emotions)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xlabel('Emotion', fontsize=14)
    plt.suptitle('Average Confusion Matrix Visualization', fontsize=18, y=0.96)

    plt.legend()

    plt.show()

    return plt

#plot_cm(average_confusion_matrix)

def conf_matrices():
    norm_cn = np.load('cm_matrices.npy', allow_pickle=True).item()

    del norm_cn['concat']

    # Add an aggregated confusion matrix for last plot
    aggregated_cm = np.mean([cm for cm in norm_cn.values()], axis=0)
    norm_cn['aggregated'] = aggregated_cm

    # Make values into percent
    for model, cm in norm_cn.items():
        cm = cm * 100
        # 2 decimals
        cm = np.round(cm, 2)
        norm_cn[model] = cm




    # Make heatmap for each model in norm_cn, plot as subplot
    fig, ax = plt.subplots(2, 3, figsize=(18, 14))

    plt.tight_layout()

    # add a bit space on sides, so plot fits
    plt.subplots_adjust(left=0.1, right=0.975, top=0.95, bottom=0.1)


    # Add gap between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.30)


    vmin = np.min([np.min(cm) for cm in norm_cn.values()])
    vmax = np.max([np.max(cm) for cm in norm_cn.values()])

    for i, (model, cm) in enumerate(norm_cn.items()):
        if model == 'aggregated':
            sns.heatmap(cm, annot=True, ax=ax[i // 3, i % 3], cmap='Greens', xticklabels=emotions, yticklabels=emotions
                        , vmin=vmin, vmax=vmax)
        else:
            sns.heatmap(cm, annot=True, ax=ax[i // 3, i % 3], cmap='Blues', xticklabels=emotions, yticklabels=emotions
                            , vmin=vmin, vmax=vmax)
        ax[i // 3, i % 3].set_title(key_map[model], fontsize=16)
        ax[i // 3, i % 3].set_xlabel('Predicted Emotion', fontsize=14)
        ax[i // 3, i % 3].set_ylabel('True Emotion', fontsize=14)

        # Rotate x labels
        ax[i // 3, i % 3].set_xticklabels(ax[i // 3, i % 3].get_xticklabels(), rotation=45, horizontalalignment='right')
        # Make y labels horizontal
        ax[i // 3, i % 3].set_yticklabels(ax[i // 3, i % 3].get_yticklabels(), rotation=0, horizontalalignment='right')


    plt.savefig('cm_heatmaps.png')
    #plt.suptitle('Normalized Confusion Matrices', fontsize=24, y=0.95)
    plt.show()

def find_common_confusions():
    norm_cn = np.load('cm_matrices.npy', allow_pickle=True).item()

    del norm_cn['concat']

    # Get average / aggregated confusion matrix
    aggregated_cm = np.mean([cm for cm in norm_cn.values()], axis=0)

    # Get the most common confusions
    common_confusions = []
    for i in range(8):
        for j in range(8):
            if i != j:
                common_confusions.append((i, j, aggregated_cm[i, j]))

    # Sort by most common
    common_confusions.sort(key=lambda x: x[2], reverse=True)


    # Print the most common confusions
    for i, j, count in common_confusions:
        print(f'{emotions[i]} -> {emotions[j]}: {count:.2f}%')

    return common_confusions



#conf_matrices()

find_common_confusions()
