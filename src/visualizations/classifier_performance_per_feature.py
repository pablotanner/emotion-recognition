"""
Generates visualization of classifier performance per feature type, used in Section 6.3.2
data was manually copied from experiment results
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Visualize the performance of a classifier per feature, classifier params were always optimized for each feature

SVC = {
    'facs':0.4375747580746948,
    'landmarks_3d': 0.48464970042099553,
    #'embedded': 0.4597618607618611,
    'embedded':0.5208743814139565,
    'hog': 0.5181242336010964,
    'pdm':0.4901139641136941
}


RandomForest = {
    'facs':0.43169672036602713,
    'landmarks_3d':0.4090411011572917,
    'embedded':0.34796128263293474,
    'hog':0.23050979069203908,
    'pdm':0.33102614927133156
}

LogisticRegression = {
    'facs':0.4197008727748087,
    'landmarks_3d':0.4417172901687877,
    #'embedded':0.3175969281007253,
    'embedded': 0.40664819000713853,
    'hog':0.4010136692851657,
    'pdm':0.3096157979749458
}

MLP = {
    'facs':0.4260352185781845,
    'landmarks_3d':0.44219983852780487,
    #'embedded': 0.4574212007089776,
    'embedded': 0.5253797828202786,
    'hog':0.5161920541196683,
    'pdm':0.46269635380339574
}

SequentialNN = {
    'facs':0.43244646084047833,
    'landmarks_3d':0.449358125639948,
    #'embedded':0.46010851165096506,
    'embedded':0.5220851165096506,
    'hog':0.5299485069471201,
    'pdm':0.4649903144338119
}

# Create a dataframe
data = {
    #'SVC': SVC,
    #'ProbaSVC': ProbaSVC,
    'SVC': SVC,
    #'LinearSVC': LinearSVC,
    'RandomForest': RandomForest,
    'LogisticRegression': LogisticRegression,
    'MLP': MLP,
    'SequentialNN': SequentialNN
}

# Rename the feature names to be more readable
feature_names = {
    'facs': 'FAUs',
    'landmarks_3d': '3D Landmarks',
    'embedded': 'Embeddings',
    'hog': 'HOG',
    'pdm': 'PDM'
}

df = pd.DataFrame(data)

# Convert to percentage
df = df * 100

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f")

# Rename y labels to readable
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], [feature_names[feature] for feature in df.index], rotation=0)

# Make the plot readable
plt.xticks(rotation=35, fontsize=12)
plt.yticks(rotation=0, fontsize=14)
plt.title('Classifier Balanced Accuracy per Feature Type (in %)', fontsize=16, y=1)

# Remove margin on top
#plt.subplots_adjust(top=0.93, bottom=0.2,right=1, left=0.20)

plt.axis('equal')
plt.tight_layout()

# Increase resolution
plt.savefig('classifier_performance_per_feature.pdf')



plt.show()
