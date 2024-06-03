# Visualize the relative gain per added classifier for our score fusion via stacking
from matplotlib import pyplot as plt

# The models, a subset starting with i=1 (only hog) and ending with i=5 (all models) are used to train a stacking classifier
models = ['HOG', 'PDM', '3D Landmarks', 'Embeddings', 'FAUs']

# The relative gain in accuracy for each added classifier, starting with only hog and ending with all
test_accuracies = [0.512594131394443, 0.5383017398078421, 0.5466112698000519, 0.5476499610490781, 0.5528434172942093]
# Same as above but uses the validation set bal accs, just for comparison
#val_bal_acc = [0.5216863128236839,0.5532726368952153,0.5564732083637857, 0.5680217733510686,0.56829543150788]


# Calculate the percent gain and absolute gain in accuracy for each added classifier
percent_gains = [round((test_accuracies[i] - test_accuracies[i-1]) * 100 / test_accuracies[i-1], 2) for i in range(1, len(test_accuracies))]
absolute_gains = [round((test_accuracies[i] - test_accuracies[i-1]) * 100, 2) for i in range(1, len(test_accuracies))]


# Create subplots (next to each other)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot percent gain in the first subplot (skip the first model)
bars1 = ax1.bar(models[1:], percent_gains, color='lightskyblue')
# Annotate the bars with the percent gain
for bar, accuracy in zip(bars1, test_accuracies[1:]):
    height = bar.get_height()

ax1.set_ylabel('Percent Gain in Balanced Accuracy (%)', fontsize=14)
ax1.set_xlabel('Last Added Classifier', fontsize=14)
#ax1.set_title('Percent Gain in Balanced Accuracy for Stacking Classifier', fontsize=16)
ax1.set_ylim(0, max(percent_gains) + 2)
ax1.grid(True, dashes=(5, 5))

# Plot absolute accuracy values in the second subplot
bars2 = ax2.bar(models, [acc * 100 for acc in test_accuracies], color='palegreen')
for bar, accuracy in zip(bars2, test_accuracies):
    height = bar.get_height()
    ax2.annotate(f'{accuracy * 100:.2f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom')
ax2.set_ylabel('Balanced Accuracy (%)', fontsize=14)
ax2.set_xlabel('Last Added Classifier', fontsize=14)
#ax2.set_title('Absolute Balanced Accuracy for Stacking Classifier', fontsize=16)
ax2.set_ylim(50, max([acc * 100 for acc in test_accuracies]) + 5)
ax2.grid(True, dashes=(5, 5))

# Adjust layout
plt.tight_layout()
plt.savefig('relative_gain_stacking.png')
plt.show()
