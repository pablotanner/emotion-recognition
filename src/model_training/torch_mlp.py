import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import optim


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, num_classes, num_epochs=10, batch_size=32, learning_rate=0.001,
                 class_weight=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.model = self._build_model()

        if self.class_weight is not None:
            weight_list = [self.class_weight[i] for i in range(self.num_classes)]
            weight_tensor = torch.tensor(weight_list, dtype=torch.float32).cuda()
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        return model

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        y_tensor = torch.tensor(y, dtype=torch.long).cuda()
        self.model = self.model.cuda()
        self.model.train()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return torch.softmax(outputs, dim=1).cpu().numpy()