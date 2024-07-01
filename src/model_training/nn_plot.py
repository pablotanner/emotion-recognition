from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam



class NeuralNetworkKeras:
    def __init__(self, input_dim, class_weight=None, num_epochs=10, batch_size=32, learning_rate=0.001, verbose=0):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(8)  # No activation function for final layer (equivalent to nn.Linear in PyTorch)
        ])

        if class_weight is not None:
            class_weight_dict = {i: class_weight[i] for i in range(len(class_weight))}
            self.class_weight = class_weight_dict
        else:
            self.class_weight = None

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train,
                                      epochs=self.num_epochs,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      class_weight=self.class_weight)

    def plot_model(self, filename='model_plot.png'):
        from keras.utils import plot_model
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

    def summary(self):
        self.model.summary()



# Plot model
nn = NeuralNetworkKeras(input_dim=300)
nn.summary()
nn.plot_model('model_plot.png')



