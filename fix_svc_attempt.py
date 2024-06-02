import numpy as np
from sklearn.metrics import balanced_accuracy_score
from src.model_training import SVC

X_train = np.load('train_facs_features.npy').astype(np.float32)
y_train = np.load('y_train.npy')
X_val = np.load('val_facs_features.npy').astype(np.float32)
y_val = np.load('y_val.npy')
X_test = np.load('test_facs_features.npy').astype(np.float32)
y_test = np.load('y_test.npy')

svc = SVC(C=1, probability=True, kernel='rbf', class_weight='balanced')

svc.fit(X_train, y_train)

balanced_accuracy = balanced_accuracy_score(y_val, svc.predict(X_val))

print(f"Balanced Accuracy on Validation: {balanced_accuracy}")

