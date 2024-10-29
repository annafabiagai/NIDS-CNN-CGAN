import os
import numpy as np
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.backend import sum
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

##################################################################################
# Data Preparation - Load Images from Folder
##################################################################################

def load_images_from_folder(folder):
    images = []
    labels = []

    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img)
            images.append(img_array)

            # Extract label from filename
            label = filename.split('_')[-1].split('.')[0]
            labels.append(label)

    return np.array(images), np.array(labels)

# Load images and labels from folder
folder = '/Users/annafabia/Desktop/NIDS_deep_images/classifier/smote_class_train_images'
images, labels = load_images_from_folder(folder)

# Convert string labels to integers
def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    return integer_encoded, label_encoder.classes_

# Encode labels
labels_int, class_names = encode_labels(labels)


# Reshape images for compatibility with the network architecture
images = expand_dims(images, axis=-1)  # Add channel dimension for grayscale
images = (images.astype('float32') - 127.5) / 127.5  # Normalize to [-1, 1]

# Split into train and test sets
train_images, test_images, train_labels_int, test_labels_int = train_test_split(images, labels_int, test_size=0.2, random_state=42)



# One-hot encode labels
from keras.utils import to_categorical
train_labels_one_hot = to_categorical(train_labels_int)
test_labels_one_hot = to_categorical(test_labels_int)

# Combine images and integer labels into a dataset tuple
dataset = (train_images, train_labels_int)

# Function to generate latent points for the generator
def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

##################################################################################
# Model Definitions
##################################################################################

# Define the standalone generator model
from keras.layers import BatchNormalization

def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * 7 * 10  # Adjust the number of nodes to match the desired output dimensions
    X = Dense(n_nodes)(in_lat)
    X = LeakyReLU(alpha=0.2)(X)
    X = Reshape((7, 10, 128))(X)  # Start with a size of (7, 10, 128)
    X = Conv2DTranspose(128, (3,3), padding='same')(X)  # (7, 10, 128)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv2DTranspose(64, (3,3), padding='same')(X)  # (7, 10, 64)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    out_layer = Conv2DTranspose(1, (3,3), activation='tanh', padding='same')(X)  # (7, 10, 1)

    model = Model(in_lat, out_layer)
    return model


def define_discriminator(in_shape=(7, 10, 1), n_classes=15):
    in_image = Input(shape=in_shape)
    X = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
    X = LeakyReLU(alpha=0.2)(X)

    X = Conv2D(64, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)

    X = Conv2D(128, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)

    X = Flatten()(X)
    X = Dropout(0.2)(X)
    X = Dense(n_classes)(X)

    model = Model(inputs=in_image, outputs=X)
    return model

# Define the supervised discriminator model
def define_sup_discriminator(disc):
    model = Sequential()
    model.add(disc)
    model.add(Activation('softmax'))
    model.compile(optimizer=Adam(learning_rate=0.00005, beta_1=0.5),
                  loss="categorical_crossentropy", metrics=['accuracy'])
    return model

# Custom activation function for the unsupervised discriminator
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x / (Z_x + 1)
    return D_x

# Define the unsupervised discriminator model
def define_unsup_discriminator(disc):
    model = Sequential()
    model.add(disc)
    model.add(Lambda(custom_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005, beta_1=0.5))
    return model

# Define the combined generator and discriminator model (for updating the generator)
def define_gan(gen_model, disc_unsup):
    disc_unsup.trainable = False
    gan_output = disc_unsup(gen_model.output)
    model = Model(gen_model.input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005, beta_1=0.5))
    return model

##################################################################################
# Training
##################################################################################

def select_supervised_samples(dataset, n_samples=150, n_classes=15):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = 20  # Adjust the number of samples per class
    for i in range(n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    # Convert y_array to one-hot encoded labels
    y_array_one_hot = to_categorical(y_array, num_classes=n_classes)

    return X_array, y_array_one_hot


# Function to generate real samples for supervised training
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples, 1))  # Labels for real images
    return [X, labels], y

# Function to generate fake samples using the generator
def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    fake_images = generator.predict(z_input)
    y = zeros((n_samples, 1))  # Labels for fake images
    return fake_images, y

# Function to summarize and save model performance
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset, n_samples=150):
    X, _ = generate_fake_samples(gen_model, latent_dim, n_samples)
    X = (X + 1) / 2.0  # Scale to [0, 1] for plotting
    for i in range(100):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(X[i, :, :, 0], cmap='gray_r')
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    plt.savefig(filename1)
    plt.close()

    X, y = dataset
    y_one_hot = to_categorical(y, num_classes=15)
    _, acc = disc_sup.evaluate(X, y_one_hot, verbose=0)
    print('Discriminator Accuracy: %.3f%%' % (acc * 100))

    filename2 = 'gen_model_%04d.h5' % (step + 1)
    gen_model.save(filename2)

    filename3 = 'disc_sup_%04d.h5' % (step + 1)
    disc_sup.save(filename3)

    print('>Saved: %s, %s, and %s' % (filename1, filename2, filename3))

# Function to train the SGAN
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=20, n_batch=100):
    # Select supervised samples
    X_sup, y_sup = select_supervised_samples(dataset)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)

    # Initialize lists to store loss and accuracy
    sup_loss_list, sup_acc_list = [], []
    d_loss_real_list, d_loss_fake_list = [], []
    gan_loss_list = []

    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))

    for i in range(n_steps):
        # Generate supervised real samples
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)
        print(sup_loss, sup_acc)
        sup_loss = sup_loss.item()
        sup_acc = sup_acc.item()


        # Generate unsupervised real samples
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss_real = disc_unsup.train_on_batch(X_real, y_real)
        d_loss_real = d_loss_real.item()



        # Generate fake samples using the generator
        X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
        d_loss_fake = disc_unsup.train_on_batch(X_fake, y_fake)
        d_loss_fake = d_loss_fake.item()
 



        # Train the generator
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        gan_loss = gan_model.train_on_batch(X_gan, y_gan)
        if isinstance(gan_loss, list):
            gan_loss = gan_loss[0]
            gan_loss= gan_loss.item()


        # Store loss and accuracy
        sup_loss_list.append(sup_loss)
        sup_acc_list.append(sup_acc)
        d_loss_real_list.append(d_loss_real)
        d_loss_fake_list.append(d_loss_fake)
        gan_loss_list.append(gan_loss)

        # Print progress
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, sup_loss, sup_acc*100, d_loss_real, d_loss_fake, gan_loss))

        # Evaluate performance every epoch
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, gen_model, disc_sup, latent_dim, dataset)




##################################################################################
# Main Execution
##################################################################################
latent_dim = 100

# Create the discriminator models
disc = define_discriminator()
disc_sup = define_sup_discriminator(disc)
disc_unsup = define_unsup_discriminator(disc)

# Define and compile the generator and GAN model
gen_model = define_generator(latent_dim)
gan_model = define_gan(gen_model, disc_unsup)


# Train the SGAN model
train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=20, n_batch=100)


