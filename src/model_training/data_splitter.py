from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, X, y, test_size: float = 0.2):
        self.X = X
        self.y = y
        self.test_size = test_size

    def split_data(self) -> tuple:
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

    @staticmethod
    def scale_data(X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler.transform(X_train), scaler.transform(X_test)
