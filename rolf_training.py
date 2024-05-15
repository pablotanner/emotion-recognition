import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from src.model_training.neural_network import NeuralNetwork

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("Loading data...")
    X_train = np.load("X_train.npy")
    X_val = np.load("X_val.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    y_test = np.load("y_test.npy")

    logger.info("Data loaded.")

    # Calculate class weights using sklearn
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model
    logger.info("Initializing model")
    input_dim = X_train.shape[1]
    model = NeuralNetwork(input_dim)

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0

    logger.info("Training model...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Early stopping
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the model if it has the best validation loss so far
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            trigger_times += 1
            print(f'Early stopping trigger times: {trigger_times}')

            if trigger_times >= patience:
                print('Early stopping!')
                break

    # Step 6: Testing Loop (Optional)
    logger.info("Testing model...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%")