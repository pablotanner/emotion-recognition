
from src.ensemble import RandomForestClassifier
import numpy as np
# Try out custom RF classifier with GPU support


# Create a random dataset
X = np.random.rand(10000, 10)
y = np.random.randint(0, 2, 10000)


# Create a custom RF classifier with GPU support
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')

# Fit the classifier
clf.fit(X, y)

# Predict
y_pred = clf.predict(X)

# Evaluate
accuracy = np.mean(y == y_pred)
