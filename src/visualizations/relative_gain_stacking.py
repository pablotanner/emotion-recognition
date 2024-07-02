# Visualize the relative gain per added classifier for our score fusion via stacking
from matplotlib import pyplot as plt

from color_mapping import color_mappings

INCLUDE_CONCAT = True

classifiers = {
    'HOG': 'MLP',
    'Embeddings': 'MLP',
    'PDM': 'SVC',
    '3D Landmarks': 'SVC',
    'FAUs': 'MLP',
    'Mixed': 'MLP'
}

# The models, a subset starting with i=1 (only hog) and ending with i=5 (all models) are used to train a stacking classifier
models = ['HOG', 'Embeddings', 'PDM', 'FAUs'] # HOG, PDM, Embedded Facs

# The relative gain in accuracy for each added classifier, starting with only hog and ending with all
test_accuracies = [0.5040249285899766, 0.5333679563749676, 0.5447935601142561, 0.549467670734874]

if INCLUDE_CONCAT:
    models = ['Mixed', 'Embeddings','3D Landmarks', 'HOG', 'FAUs']
    test_accuracies = [0.5627109841599585,  0.5642690210334977, 0.5679044404050896, 0.5684237860296028, 0.5689431316541158]
# Same as above but uses the validation set bal accs, just for comparison
#val_bal_acc = [0.5216863128236839,0.5532726368952153,0.5564732083637857, 0.5680217733510686,0.56829543150788]


# Calculate the percent gain and absolute gain in accuracy for each added classifier
percent_gains = [round((test_accuracies[i] - test_accuracies[i-1]) * 100 / test_accuracies[i-1], 2) for i in range(1, len(test_accuracies))]
absolute_gains = [round((test_accuracies[i] - test_accuracies[i-1]) * 100, 2) for i in range(1, len(test_accuracies))]


# Create subplots (next to each other)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot percent gain in the first subplot (skip the first model)
bars1 = ax1.bar(models[1:], percent_gains, color='lightskyblue')
# Set bar color according to the classifier
for i, bar in enumerate(bars1):
    bar.set_color(color_mappings[classifiers[models[i+1]]])
    bar.set_label(classifiers[models[i+1]])

# Annotate the bars with the percent gain
for bar, accuracy in zip(bars1, test_accuracies[1:]):
    height = bar.get_height()

ax1.set_ylabel('Percent Gain in Balanced Accuracy (%)', fontsize=18)
ax1.set_xlabel('Last Added Feature Type', fontsize=18)

# Increase bar label fontsize
ax1.tick_params(axis='both', which='major', labelsize=14)

#ax1.set_title('Percent Gain in Balanced Accuracy for Stacking Classifier', fontsize=16)
ax1.set_ylim(0, max(percent_gains) + 2)
ax1.grid(True, dashes=(5, 5))

# Set vlim from 0 to 8
ax1.set_ylim(0, 7)

# Plot absolute accuracy values in the second subplot
bars2 = ax2.bar(models, [acc * 100 for acc in test_accuracies], color='palegreen')
for bar, accuracy in zip(bars2, test_accuracies):
    height = bar.get_height()
    ax2.annotate(f'{accuracy * 100:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points", fontsize=14,
                 ha='center', va='bottom')

# Set bar color according to the classifier
for i, bar in enumerate(bars2):
    bar.set_color(color_mappings[classifiers[models[i]]])
    bar.set_label(classifiers[models[i]])



# Increase bar label fontsize
ax2.tick_params(axis='both', which='major', labelsize=14)


ax2.set_ylabel('Balanced Accuracy (%)', fontsize=18)
ax2.set_xlabel('Last Added Feature Type', fontsize=18)
#ax2.set_title('Absolute Balanced Accuracy for Stacking Classifier', fontsize=16)
ax2.set_ylim(50, 60)
ax2.grid(True, dashes=(5, 5))



# Add legend for all classifier colors
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)


# Adjust layout
plt.tight_layout()
if INCLUDE_CONCAT:
    plt.savefig('relative_gain_stacking_concat.pdf')
else:
    plt.savefig('relative_gain_stacking.pdf')
plt.show()
