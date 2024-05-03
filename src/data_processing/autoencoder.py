from keras.layers import Input, Dense
from keras.models import Model

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = Model(input_layer, decoder)

    encoder_model = Model(input_layer, encoder)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder_model = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder_model, decoder_model


def reduce_dimensionality(data, encoding_dim):
    autoencoder, encoder_model, decoder_model = build_autoencoder(data.shape[1], encoding_dim)
    autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)
    return encoder_model.predict(data)

