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

# Contains all except facs and pdm
confusion_matrices = np.load('cm.npy', allow_pickle=True).item()

cm_rest = np.load('cm_matrices_not_normalized.npy', allow_pickle=True).item()

confusion_matrices['facs'] = cm_rest['facs']
confusion_matrices['pdm'] = cm_rest['pdm']

del cm_rest
del confusion_matrices['concatenated']

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

def single_cm_heatmap(cm, name, color='Blues'):
    cm = cm * 100
    cm = np.round(cm, 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap=color, square=True, xticklabels=emotions, yticklabels=emotions,cbar_kws={'shrink': 1},
                #vmin=0, vmax=80,
                )
    plt.xlabel('Predicted Emotion', fontsize=16, labelpad=15)
    plt.ylabel('True Emotion', fontsize=16, labelpad=15)



    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    #plt.savefig('facs_cm.png')
    plt.tight_layout()
    plt.savefig(f'confusion_matrices/{name}_cm.pdf')
    plt.show()

def find_common_confusions(matrix):
    common_confusions = []
    for i in range(8):
        for j in range(8):
            if i != j:
                common_confusions.append((i, j, matrix[i, j]))

    # Sort by most common
    common_confusions.sort(key=lambda x: x[2], reverse=True)


    # Print the most common confusions (top 10)
    c = 0
    for i, j, count in common_confusions:
        c += 1
        print(f'{emotions[i]} - {emotions[j]}: {count:.2f}%')
        if c == 5:
            break

    return common_confusions

def two_matrices_heatmap(matrix1, matrix2):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Add gap between subplots
    plt.subplots_adjust(hspace=0.35, wspace=0.30)

    # Use same vmin and vmax for both plots
    vmin = np.min([np.min(matrix1), np.min(matrix2)])
    vmax = np.max([np.max(matrix1), np.max(matrix2)])

    sns.heatmap(matrix1, annot=True, ax=ax[0], cmap='Purples', xticklabels=emotions, yticklabels=emotions,
                vmin=vmin, vmax=vmax
                )
    ax[0].set_title('Concatenated', fontsize=16)
    ax[0].set_xlabel('Predicted Emotion', fontsize=14)
    ax[0].set_ylabel('True Emotion', fontsize=14)

    sns.heatmap(matrix2, annot=True, ax=ax[1], cmap='Purples', xticklabels=emotions, yticklabels=emotions,
                vmin=vmin, vmax=vmax)
    ax[1].set_title('Stacking Classifier', fontsize=16)
    ax[1].set_xlabel('Predicted Emotion', fontsize=14)
    ax[1].set_ylabel('True Emotion', fontsize=14)

    plt.savefig('cm_stacking_concat.png')
    plt.show()

#conf_matrices()

#find_common_confusions()

def get_matrices():
    # Get Aggregated Confusion Matrix
    norm_cn = np.load('cm_matrices.npy', allow_pickle=True).item()
    del norm_cn['concat']
    # Get average / aggregated confusion matrix
    aggregated_cm = np.mean([cm for cm in norm_cn.values()], axis=0) * 100

    # Get Concatenated Confusion Matrix
    concat_matrix = np.load('cm_concat.npy')
    # Normalize
    concat_matrix = concat_matrix / np.sum(concat_matrix, axis=1)[:, np.newaxis] * 100
    # rounded
    concat_matrix = np.round(concat_matrix, 2)

    stacked_matrix = np.load('cm_stacking.npy')
    stacked_matrix = stacked_matrix / np.sum(stacked_matrix, axis=1)[:, np.newaxis] * 100
    stacked_matrix = np.round(stacked_matrix, 2)


    return aggregated_cm, concat_matrix, stacked_matrix

aggregated_cm, concat_matrix, stacked_matrix = get_matrices()



#single_cm_heatmap(stacked_matrix)

#two_matrices_heatmap(concat_matrix, stacked_matrix)
norm_cn = np.load('cm_matrices.npy', allow_pickle=True).item()

norm_cn['embedded'] = np.load('embeddings_cm_normalized.npy')
del norm_cn['concat']

mean_matrix = np.mean([cm for cm in norm_cn.values()], axis=0)
std_matrix = np.std([cm for cm in norm_cn.values()], axis=0)

"""
for model, cm in norm_cn.items():
    single_cm_heatmap(cm, model)

mean_matrix = np.mean([cm for cm in norm_cn.values()], axis=0)
std_matrix = np.std([cm for cm in norm_cn.values()], axis=0)

single_cm_heatmap(mean_matrix, 'mean', color='Greens')
single_cm_heatmap(std_matrix, 'std', color='Reds')

"""



print('Aggregated')
find_common_confusions(mean_matrix * 100)
print(20 * '-')

for model, cm in norm_cn.items():
    # Values in percent
    cm = cm * 100
    print(key_map[model])
    find_common_confusions(cm)
    print(20 * '-')

#print('Stacked')
#find_common_confusions(stacked_matrix)
#print(20 * '-')

print('Concatenated')
find_common_confusions(concat_matrix)
print(20 * '-')