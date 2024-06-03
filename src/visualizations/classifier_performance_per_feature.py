import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visualize the performance of a classifier per feature, classifier params were always optimized for each feature

NewSVC = {
    'facs':0.4375747580746948,
    'landmarks_3d': 0.48464970042099553,
    'embedded': 0.4597618607618611,
    'hog': 0.5181242336010964,
    'pdm':0.4901139641136941
}

SVC = {
    'facs':0.43858737723542457,
    'landmarks_3d':0.47852156857211503,
    'embedded':0.45714285714285713,
    'hog': 0.5181242336010964,
    'pdm':0.4887116144381338
}

LinearSVC = {
    'facs':0.4167163289630512,
    'landmarks_3d':0.42795282402245377,
    'embedded':0.4302276189389127,
    'hog':0.5011751608059447,
    'pdm':0.426429374571
}

ProbaSVC = {
    'facs':0.2960017958081935,
    'landmarks_3d':0.29660846986469863,
    'embedded':0.40380037930425716,
    'hog': 0.4130294820198,
    'pdm':0.3317595501334409
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
    'embedded':0.3175969281007253,
    'hog':0.4010136692851657,
    'pdm':0.3096157979749458
}

MLP = {
    'facs':0.4260352185781845,
    'landmarks_3d':0.44219983852780487,
    'embedded': 0.4574212007089776,
    'hog':0.5161920541196683,
    'pdm':0.46269635380339574
}

SequentialNN = {
    'facs':0.43244646084047833,
    'landmarks_3d':0.449358125639948,
    'embedded':0.46010851165096506,
    'hog':0.5299485069471201,
    'pdm':0.4649903144338119
}

# Create a dataframe
data = {
    #'SVC': SVC,
    #'ProbaSVC': ProbaSVC,
    'SVC': NewSVC,
    'LinearSVC': LinearSVC,
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
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Remove margin on top
plt.subplots_adjust(top=0.93, bottom=0.15,right=1)

# Increase resolution
plt.title('Classifier Balanced Accuracy per Feature (in %)')
plt.savefig('classifier_performance_per_feature.png', dpi=300)



plt.show()
