
# For the experiment, where I try to find what is more important for score fusion with probability stacking,
# the classifier (train different classifiers with the same feature) or the feature (train the same classifier with different features)
# This code is to visualize it


import matplotlib.pyplot as plt
import numpy as np

# Without normal SVC because it can't be used for probability stacking
feature_data = {
    'FACS': {
        #'ProbaSVC': 0.30082937922322794,
        'SVC': 0.4394365847036077,
        'LinearSVC': 0.4150818239034705,
        'RandomForest': 0.40844026644235015,
        'LogisticRegression':0.30226751883669045,
        'MLP':  0.4280072305310446,
        'NN': 0.4190464153841033,
        'Stacking': 0.45040617848136666
    },
    'Landmarks': {
        #'ProbaSVC': 0.2913596826796941,
        'SVC': 0.46925351977430735,
        'LinearSVC':  0.424583889396807,
        'RandomForest':  0.37033860934241297,
        'LogisticRegression': 0.31301364049721,
        'MLP': 0.45813919072802145,
        'NN':  0.46607523272653656,
        'Stacking': 0.4837351634008407
    },
    'PDM': {
        #'ProbaSVC': 0.3124604609149185,
        'SVC': 0.4764234030931595,
        'LinearSVC': 0.4189193363161993,
        'RandomForest': 0.36123012095717233,
        'LogisticRegression': 0.3075308513345376,
        'MLP':  0.4568980638844722,
        'NN': 0.4541749754958806,
        'Stacking':  0.4970049178109605
    },
    'Embedded': {
        #'ProbaSVC': 0.40424141346273734,
        'SVC':  0.4105213989055462,
        'LinearSVC': 0.32592574767527055,
        'RandomForest': 0.2621200755547471,
        'LogisticRegression': 0.224910967905037,
        'MLP': 0.3559787116934988,
        'NN':  0.34620306254498556,
        'Stacking': 0.40930195203108055
    },
    'HOG': {
        #'ProbaSVC': None,
        'SVC': 0.512958022772099,
        'LinearSVC': 0.4574975217516636,
        'RandomForest': 0.3960854499665586,
        'LogisticRegression': 0.34803428823339766,
        'MLP':  0.47336814939202226,
        'NN': 0.49594010333142524,
        'Stacking': 0.5218187024820742
    },
}

classifier_data = {
    'LogisticRegression': {
        'HOG': 0.34803428823339766,
        'Landmarks': 0.31301364049721,
        'PDM': 0.3075308513345376,
        'FACS': 0.30226751883669045,
        'Embedded':  0.224910967905037,
        'Stacking': 0.4792491762107507,
    },
    'MLP': {
        'HOG': 0.47336814939202226,
        'Landmarks': 0.45813919072802145,
        'PDM': 0.4568980638844722,
        'FACS':  0.4280072305310446,
        'Embedded': 0.3559787116934988,
        'Stacking':0.5140910380375265,
    },
    'NN': {
        'HOG':  0.49594010333142524,
        'Landmarks':  0.46607523272653656,
        'PDM': 0.4541749754958806,
        'FACS':  0.4190464153841033,
        'Embedded': 0.34620306254498556,
        'Stacking':  0.5333450963924531,
    },
    'LinearSVC': {
        'HOG': 0.4574975217516636,
        'Landmarks': 0.424583889396807,
        'PDM': 0.4189193363161993,
        'FACS': 0.4150818239034705,
        'Embedded': 0.32592574767527055,
        'Stacking': 0.4911323063566714,
    },
    'RandomForest': {
        'HOG': 0.3960854499665586,
        'Landmarks':0.37033860934241297,
        'PDM':  0.36123012095717233,
        'FACS': 0.40844026644235015,
        'Embedded':  0.2621200755547471,
        'Stacking':  0.5056573506483925,
    },
    'SVC': {
        'HOG': 0.512958022772099,
        'Landmarks':  0.46925351977430735,
        'PDM': 0.4764234030931595,
        'FACS': 0.4394365847036077,
        'Embedded': 0.4105213989055462,
        'Stacking': 0.5449414775358494,

    },
}


def plot_grouped_bar_chart(data, title, x_label):
    # Prepare data
    labels = list(data.keys())
    sub_labels = [key for key in data[labels[0]].keys() if key != 'Stacking']
    stacking_values = [data[label]['Stacking'] * 100 for label in labels]
    other_values = np.array([[data[label][sub_label] * 100 for sub_label in sub_labels] for label in labels])

    x = np.arange(len(labels))  # label locations
    stacking_width = 0.2  # width of the Stacking bars
    other_width = 0.1  # width of the other bars
    gap = 0.05  # gap between the Stacking bar and the other bars

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color palette
    colors = plt.get_cmap('tab20c')

    # Plot Stacking bars
    ax.bar(x, stacking_values, stacking_width, label='Stacking', color=colors(0))

    # Plot other bars
    for i in range(len(sub_labels)):
        ax.bar(x + stacking_width / 2 + gap + other_width * i, other_values[:, i], other_width, label=sub_labels[i],
               color=colors(i + 1))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel('Balanced Accuracy (%)', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x + stacking_width / 2 + gap + other_width * ((len(sub_labels) - 1) / 2))
    ax.set_xticklabels(labels)
    ax.legend()

    # Rotate x labels
    plt.xticks(rotation=45)

    # legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    # Adjust space below plots and overall layout
    plt.subplots_adjust(bottom=0.3, right=0.8)
    plt.ylim(0, 60)

    # Save the plot
    plt.savefig(f'{title}_stacking_comparison.png')

    plt.show()


# Plot for feature data
plot_grouped_bar_chart(feature_data, 'Stacking Accuracy per Feature', 'Feature')

# Plot for classifier data
plot_grouped_bar_chart(classifier_data, 'Stacking Accuracy per Classifier', 'Classifier')