from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib  # For saving the best model parameters

def run_grid_search(X_train, y_train):

    models_params = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVC': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [200, 300, 400]
            }
        }
    }

    results = []

    for model_name, mp in models_params.items():
        grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, verbose=2, n_jobs=-1, scoring='balanced_accuracy')
        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_
        results.append({
            'model': model_name,
            'best_score': best_score,
            'best_params': best_parameters
        })
        try:
            # Optionally save the best model parameters
            joblib.dump(grid_search.best_estimator_, f'../../models/{model_name}_best_params.pkl')
        except:
            print(f"Error saving {model_name} best parameters")

    for result in results:
        print(f"{result['model']} best score: {result['best_score']}")
        print(f"{result['model']} best parameters: {result['best_params']}")
    return results
