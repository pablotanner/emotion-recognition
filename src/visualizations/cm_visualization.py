import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load normalized confusion matrices for each model/feature
#confusion_matrices = np.load('cm_matrices.npy', allow_pickle=True).item()
confusion_matrices = np.load('cm_matrices_not_normalized.npy', allow_pickle=True).item()
emotions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

# Get averages for all models except for FACS
average_confusion_matrices = {}
for model, matrix in confusion_matrices.items():
    if model != 'facs':
        average_confusion_matrices[model] = matrix

# Get the average confusion matrix
average_confusion_matrix = np.round(np.mean([matrix for matrix in average_confusion_matrices.values()], axis=0))

# Facs confusion matrix
facs_confusion_matrix = confusion_matrices['facs']


facs_alt_confusion_matrix = np.array([
[217,  14,  56,  39,  23,  16,  51,  62],
[10, 329,  14,  20,   4,  12,   9,  83],
[106,  20, 188,  23,  27,  37,  59,  26],
[62,  45,  30, 206,  85,  20,  22,  13],
[43,  15,  37,  96, 207,  31,  36,   8],
[44,  27,  54,  38,  23, 179,  93,  23],
[86,   9,  49,  26,  33,  70, 188,  18],
[102,  92,  27,  21,   5,  20,  45, 178]])

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
    plt.suptitle('Alternative Confusion Matrix Visualization', fontsize=18, y=0.96)
    plt.title('Averaged confusion matrix for all models except FAUs', fontsize=14, y=1.02)

    plt.legend()

    plt.show()

    return plt


#plot_cm(average_confusion_matrix)
plot_cm(facs_confusion_matrix)




