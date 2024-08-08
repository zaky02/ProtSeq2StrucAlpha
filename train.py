import json
import torch
import tensorflow as tf
from model import create_encoder, create_decoder

# Load hyperparameters from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

input_dim = config["input_dim"]
output_dim = config["output_dim"]
embedding_dim = config["embedding_dim"]
units = config["units"]
batch_size = config["batch_size"]
epochs = config["epochs"]

# Load preprocessed data
data = torch.load('preprocessed_data.pth')
esm_sequences = data['esm_sequences']
masked_structural_sequences = data['masked_structural_sequences']
target_structural_sequences = data['target_structural_sequences']

# Create models
encoder = create_encoder(input_dim, embedding_dim, units)
decoder = create_decoder(output_dim, embedding_dim, units)

# Loss function and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = tf.keras.optimizers.Adam()

# Convert to TensorFlow Dataset
def create_dataset(esm_sequences, masked_structural_sequences, target_structural_sequences):
    def gen():
        for esm_seq, masked_struct_seq, target_struct_seq in zip(esm_sequences, masked_structural_sequences, target_structural_sequences):
            yield (esm_seq, masked_struct_seq), target_struct_seq

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.int32), tf.TensorSpec(shape=(None,), dtype=tf.int32)),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).padded_batch(batch_size, padded_shapes=(([None], [None]), [None]))

dataset = create_dataset(esm_sequences, masked_structural_sequences, target_structural_sequences)

# Training step
@tf.function
def train_step(input_seq, target_seq, encoder, decoder, optimizer, loss_object):
    loss = 0
    with tf.GradientTape() as tape:
        encoder_input, decoder_input = input_seq
        encoder_outputs, states = encoder(encoder_input)
        predictions = decoder(decoder_input, initial_state=states)
        loss = tf.reduce_mean(loss_object(target_seq, predictions))

    gradients = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables))
    
    return loss

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for (batch, (input_seq, target_seq)) in enumerate(dataset):
        batch_loss = train_step(input_seq, target_seq, encoder, decoder, optimizer, loss_object)
        total_loss += batch_loss
        
    print(f'Epoch {epoch+1} Loss {total_loss / (batch+1)}')
