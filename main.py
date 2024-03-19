import numpy as np
try:
    import keras
except ModuleNotFoundError:
    print("Keras is not installed. Please install Keras using 'pip install keras'")
    exit()

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import layers 

# Load the MNIST dataset and preprocess it
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Print the shapes of the training and testing datasets
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Define the dimension of the encoded representation
encoding_dim = 32

# Define the input layer for the autoencoder model
input_img = keras.Input(shape=(784,))

# Define the encoding layer
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# Define the decoding layer
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Create the autoencoder model
autoencoder = keras.Model(input_img, decoded)

# Create the encoder model
encoder = keras.Model(input_img, encoded)

# Define the input layer for the decoder model
encoded_input = keras.Input(shape=(encoding_dim,))

# Retrieve the decoding layer from the autoencoder model
decoded_layer = autoencoder.layers[-1]

# Create the decoder model
decoder = keras.Model(encoded_input, decoded_layer(encoded_input))

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Fit the autoencoder model to the training data
history = autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test, x_test))

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Generate encoded representations of the test images
encoded_imgs = encoder.predict(x_test)

# Reconstruct the test images from the encoded representations
decoded_imgs = decoder.predict(encoded_imgs)

# Visualize original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
