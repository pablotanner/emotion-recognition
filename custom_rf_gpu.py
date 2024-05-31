
import numpy as np
import dask.dataframe as dd
import dask.array as da
from src.ensemble import RandomForestClassifier

# Try out custom RF classifier with GPU support


# Create a random dataset
X = np.random.rand(10000, 10).astype(np.float32)
y = np.random.randint(0, 2, 10000).astype(np.int32)

# As Dask DF


X = da.from_array(X, chunks=(1000, 10))
y = da.from_array(y, chunks=(1000,))
X = dd.from_dask_array(X)
y = dd.from_dask_array(y)



# Create a custom RF classifier with GPU support
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', verbose=)

# Fit the classifier
clf.fit(X, y)

# Predict
y_pred = clf.predict(X)

# Evaluate
accuracy = np.mean(y == y_pred)
