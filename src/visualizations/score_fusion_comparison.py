
# For the experiment, where I try to find what is more important for score fusion with probability stacking,
# the classifier (train different classifiers with the same feature) or the feature (train the same classifier with different features)
# This code is to visualize it
import matplotlib.patches as mpatches

from color_mapping import color_mappings

import matplotlib.pyplot as plt
import numpy as np

# Without normal SVC because it can't be used for probability stacking
feature_data = {
    'FAUs': {
        #'ProbaSVC': 0.30082937922322794,
        'SVC': 0.42584585952077897,
        #'LinearSVC': 0.4150818239034705,
        'RandomForest': 0.4128616860267358,
        'LogisticRegression': 0.30096522920432967,
        'MLP': 0.36715621840072116,
        'SequentialNN':0.39739461005445553,
        'Stacking':  0.4508735013673891
    },
    '3D Landmarks': {
        #'ProbaSVC': 0.2913596826796941,
        'SVC': 0.4632207187473768,
        #'LinearSVC':  0.424583889396807,
        'RandomForest':  0.38666443599315187,
        'LogisticRegression': 0.31095149603652256,
        'MLP': 0.43869856945135643,
        'SequentialNN':  0.47283163729029276,
        'Stacking': 0.48517603134472664
    },
    'PDM': {
        #'ProbaSVC': 0.3124604609149185,
        'SVC': 0.4764234030931595,
        #'LinearSVC': 0.4189193363161993,
        'RandomForest': 0.36123012095717233,
        'LogisticRegression': 0.3075308513345376,
        'MLP':  0.4568980638844722,
        'SequentialNN': 0.4541749754958806,
        'Stacking':  0.4970049178109605
    },
    'Embeddings': {
        #'ProbaSVC': 0.40424141346273734,
        'SVC': 0.4847070877137102,
        #'LinearSVC': 0.32592574767527055,
        'RandomForest':  0.35963455485814677,
        'LogisticRegression':  0.3408180291789805,
        'MLP': 0.4716420461037578,
        'SequentialNN': 0.467113413245109,
        'Stacking':  0.5004143201159311
    },
    'HOG': {
        #'ProbaSVC': None,
        'SVC': 0.5127367708797566,
        #'LinearSVC': 0.4574975217516636,
        'RandomForest': 0.40278790256496066,
        'LogisticRegression': 0.35165090557418766,
        'MLP':  0.47663243391464705,
        'SequentialNN': 0.49967367635182824,
        'Stacking': 0.5329025527781254
    },
}

classifier_data = {
    'SVC': {
        'FAUs': 0.42584585952077897,
        '3D Landmarks': 0.4632207187473768,
        'PDM': 0.4764234030931595,
        'Embeddings':  0.4847070877137102,
        'Stacking': 0.5446545817502958,
        'HOG': 0.5127367708797566,
    },
    'RandomForest': {
        'FAUs': 0.4128616860267358,
        '3D Landmarks': 0.38666443599315187,
        'PDM': 0.36123012095717233,
        'Embeddings':  0.35963455485814677,
        'Stacking': 0.5178434399062805,
        'HOG': 0.40278790256496066,
    },
    'LogisticRegression': {
        'FAUs': 0.30096522920432967,
        '3D Landmarks': 0.31095149603652256,
        'PDM': 0.3075308513345376,
        'Embeddings':  0.3408180291789805,
        'Stacking': 0.4892494286231743,
        'HOG': 0.35165090557418766,
    },
    'MLP': {
        'FAUs': 0.36715621840072116,
        '3D Landmarks': 0.43869856945135643,
        'PDM': 0.4568980638844722,
        'Embeddings': 0.4716420461037578,
        'Stacking': 0.5206920420230381,
        'HOG': 0.47336814939202226,
    },
    'SequentialNN': {
        'FAUs': 0.39739461005445553,
        '3D Landmarks':  0.47283163729029276,
        'PDM': 0.4541749754958806,
        'Embeddings':  0.467113413245109,
        'Stacking':  0.5507349927721724,
        'HOG': 0.49967367635182824,
    },

}

# Find the average relative increase the highest of the classifiers (except Stacking) to 'Stacking'

def average_relative_increase(data):
    # Get the average relative increase
    relative_increases = []
    for key in data:
        # Highest
        highest = max([data[key][classifier] for classifier in data[key] if classifier != 'Stacking'])
        relative_increases.append((data[key]['Stacking'] - highest) / highest)

    return np.mean(relative_increases)



# Plot for per feature data
def stacking_per_feature(data):
    # For key in data, one bar group. For key in data[key], one bar in the group (get color from color_mappings)
    # X axis label is feature name
    labels = list(data.keys())
    sub_labels = [key for key in data[labels[0]].keys() if key != 'Stacking']
    stacking_values = [data[label]['Stacking'] * 100 for label in labels]
    other_values = np.array([[data[label][sub_label] * 100 for sub_label in sub_labels] for label in labels])
    x = np.arange(len(labels))  # label locations
    stacking_width = 0.2  # width of the Stacking bars
    other_width = 0.1  # width of the other bars

    fig, ax = plt.subplots(figsize=(14, 7))

    # ylim 50 to 60
    plt.ylim(0, 60)

    # grid
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot Stacking bars
    ax.bar(x, stacking_values, stacking_width, label='Stacking Classifier', color='gold', edgecolor='black')
    # Plot other bars
    for i in range(len(sub_labels)):
        ax.bar(x + stacking_width + other_width * i, other_values[:, i], other_width, label=sub_labels[i],
            color=color_mappings[sub_labels[i]],
            edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Feature Type', fontsize=18)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=18)
    ax.set_xticks(x + stacking_width + other_width * ((len(sub_labels) - 1) / 2))
    ax.set_xticklabels(labels)
    ax.legend()
    # Rotate x labels
    plt.xticks(rotation=0, fontsize=16)
    plt.tight_layout()
    ax.legend(bbox_to_anchor=(1.17, 1), loc='upper right', borderaxespad=0.)
    # Show more space on right for legend
    plt.subplots_adjust(right=0.85)

    plt.savefig('stacking_per_feature.pdf')

    plt.show()

def stacking_per_classifier(data):
    patterns = ["/", "x", "o", "\\", ".", "*", "-", "O"]
    labels = list(data.keys())
    sub_labels = [key for key in data[labels[0]].keys() if key != 'Stacking']
    stacking_values = [data[label]['Stacking'] * 100 for label in labels]
    other_values = np.array([[data[label][sub_label] * 100 for sub_label in sub_labels] for label in labels])
    x = np.arange(len(labels))  # label locations
    stacking_width = 0.2  # width of the Stacking bars
    other_width = 0.1  # width of the other bars

    fig, ax = plt.subplots(figsize=(14, 7))

    # Set ylim 50 to 60
    plt.ylim(0, 60)

    # Add grid
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot Stacking bars
    ax.bar(x, stacking_values, stacking_width, label='Stacking Classifier', color='gold', edgecolor='black')

    # Plot other bars
    for i in range(len(sub_labels)):
        for j, label in enumerate(labels):
            ax.bar(x[j] + stacking_width + other_width * i, other_values[j, i], other_width,
                   label=sub_labels[i] if j == 0 else "",  # Avoid duplicate labels in legend
                   color=color_mappings[label],
                   hatch=patterns[i],
                   edgecolor='black')

    # Custom legend handles with larger patches
    handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=3*patterns[i], label=sub_labels[i],
                              linewidth=1) for i in range(len(sub_labels))]
    stacking_handle = mpatches.Patch(facecolor='gold', edgecolor='black', label='Stacking Classifier')




    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Classifier', fontsize=18)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=18)
    ax.set_xticks(x + stacking_width + other_width * ((len(sub_labels) - 1) / 2))
    ax.set_xticklabels(labels)
    ax.legend(handles=[stacking_handle] + handles, bbox_to_anchor=(1.22, 1), loc='upper right', borderaxespad=0., fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=0, fontsize=16)
    plt.tight_layout()
    # Show more space on right for legend
    plt.subplots_adjust(right=0.82)

    plt.savefig('stacking_per_classifier.pdf')

    plt.show()


stacking_per_feature(feature_data)
plt.close()
stacking_per_classifier(classifier_data)


# Plot for feature data
#plot_grouped_bar_chart(feature_data, 'Stacking Accuracy per Feature', 'Feature')

# Plot for classifier data
#plot_grouped_bar_chart(classifier_data, 'Stacking Accuracy per Classifier', 'Classifier')