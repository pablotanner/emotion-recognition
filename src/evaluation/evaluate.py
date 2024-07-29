"""
Used for evaluation in early experiments
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    balanced_accuracy_score, classification_report


def evaluate_results(y_test, y_pred):
    # Evaluation (0.42412451361867703)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")

    #print("Classification Report:\n", classification_report(y_test, y_pred))

    """
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    """

    # Generating and printing the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)