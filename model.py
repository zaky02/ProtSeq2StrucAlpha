import tensorflow as tf
from tensorflow.keras import layers, Model

def create_encoder(input_dim, embedding_dim, units):
    inputs = layers.Input(shape=(None,))
    embedding = layers.Embedding(input_dim=input_dim, output_dim=embedding_dim)(inputs)
    encoder_outputs, state_h, state_c = layers.LSTM(units, return_state=True)(embedding)
    states = [state_h, state_c]
    return Model(inputs, [encoder_outputs, states])

def create_decoder(output_dim, embedding_dim, units):
    inputs = layers.Input(shape=(None,))
    embedding = layers.Embedding(input_dim=output_dim, output_dim=embedding_dim)(inputs)
    decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(embedding, initial_state=states)
    outputs = layers.Dense(output_dim, activation='softmax')(decoder_outputs)
    return Model(inputs, outputs)
