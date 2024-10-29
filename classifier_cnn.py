import pickle
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Function to load grayscale images and labels from a folder
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

# Load grayscale images
train_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/classifier/removed_class_train_images'
test_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/classifier/removed_class_test_images'

X_train, y_train = load_images_from_folder(train_folder_path)
X_test, y_test = load_images_from_folder(test_folder_path)

# Normalize images (scale pixel values to between 0 and 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert string labels to one-hot encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_one_hot = to_categorical(y_train_encoded, num_classes)
y_test_one_hot = to_categorical(y_test_encoded, num_classes)

# Reshape input data to fit the model input shape
X_train = np.expand_dims(X_train, axis=-1)  # Shape (num_samples, 7, 10, 1)
X_test = np.expand_dims(X_test, axis=-1)    # Shape (num_samples, 7, 10, 1)

# Define the model creation function
def create_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(112, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(48, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Initialize and compile the model
input_shape = (7, 10, 1)  # Grayscale images have 1 channel
model = create_cnn_model(input_shape, num_classes)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_one_hot, epochs=30, batch_size=32, validation_split=0.2)

# Save the model
model.save('removed_class_model.keras')

# Save the label encoder
with open('removed_class_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')

# Make predictions on the test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_one_hot, axis=1)

# Print actual and predicted labels for the first 10 samples
for i in range(10):
    print(f'Image {i}: True label = {label_encoder.inverse_transform([true_classes[i]])[0]}, Predicted label = {label_encoder.inverse_transform([predicted_classes[i]])[0]}')

# Calculate and print the classification report
report = classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_)
print(report)

# Calculate overall weighted F1 score
f1 = f1_score(true_classes, predicted_classes, average='weighted')
print(f'Overall F1 Score: {f1:.4f}')

# Plot accuracy and loss
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot training & validation accuracy values
axs[0].plot(history.history['accuracy'])
axs[0].plot(history.history['val_accuracy'])
axs[0].set_title('Model accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Model loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Validation'], loc='upper left')

# Save the plots
plot_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/classifier/plot/removed_class'
if not os.path.exists(plot_folder_path):
    os.makedirs(plot_folder_path)

fig.savefig(os.path.join(plot_folder_path, 'training_history_removed_class.png'))
print("Training history plot saved.")


