import itertools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_processing.data_loader import DataLoader

data_loader = DataLoader("./data", "./data", exclude=['landmarks', 'hog'])

y = data_loader.emotions

X_dict = data_loader.features

embeddings = ['deepface', 'facenet', 'vggface', 'openface', 'sface', 'facenet512', 'arcface']


def process_embeddings(embeddings):
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    return embeddings

for embedding in embeddings:
    X_dict[embedding] = process_embeddings(X_dict[embedding])


# For each possible subset of embeddings, train a classifier and evaluate it, keeping track of the best one
best_subset = []
best_accuracy = 0


results = []

for i in range(1, len(embeddings) + 1):
    for subset in itertools.combinations(embeddings, i):
        X = np.concatenate([X_dict[embedding] for embedding in subset], axis=1)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a classifier
        clf = SVC(C=1, kernel='rbf')
        clf.fit(X_train, y_train)
        # Evaluate it
        accuracy = clf.score(X_test, y_test)
        # If it's the best so far, update best_subset and best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_subset = subset

        print(f"Subset: {subset}, Accuracy: {accuracy}")
        results.append((subset, accuracy))

print(20 * "-")
print(f"Best subset: {best_subset}, Best accuracy: {best_accuracy}")


# Create df for results
import pandas as pd

df = pd.DataFrame(results, columns=['Subset', 'Accuracy'])

# save to csv
df.to_csv("embedding_subset_results.csv", index=False)
print(df)



