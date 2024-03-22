from model.models import RandomForestModel, SVM
from model.prepare_data import prepare_data, split_data

X, y = prepare_data()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y, use_scaler=True)


# Init the models
svm = SVM(C=1.0, kernel='rbf')
random_forest = RandomForestModel(n_estimators=200, max_depth=10)

# Train the models
svm.train(X_train, y_train)
random_forest.train(X_train, y_train)

# Evaluate the models
print("SVM:")
svm.evaluate(X_test, y_test)

print("Random Forest:")
random_forest.evaluate(X_test, y_test)


# Find optimal number of trees for Random Forest
# RandomForestModel.tree_performance(X_train, X_test, y_train, y_test)