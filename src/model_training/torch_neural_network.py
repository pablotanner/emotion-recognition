import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, class_weight=None):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 8)
        self.optimizer = None
        
        if isinstance(class_weight, dict):
            class_weight = torch.tensor([class_weight[i] for i in range(len(class_weight))], dtype=torch.float32)

        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def compile(self, optimizer):
        self.optimizer = optimizer

    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        self.train()
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                 torch.tensor(y_train, dtype=torch.long))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(torch.tensor(X, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            outputs = F.softmax(self.forward(torch.tensor(X, dtype=torch.float32)), dim=1)
        return outputs.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
