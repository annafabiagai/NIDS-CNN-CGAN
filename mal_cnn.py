'''import pickle
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
train_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/attacks_classifier/malNOSMOTE_train_images'
test_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/attacks_classifier/malNOSMOTE_test_images'

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

# prima (2,2), dropout=0.2 

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

model.save('NO_SMOTE_class_model.keras')

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
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
'''

import pickle
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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
train_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/attacks_classifier/malNOSMOTE_train_images'
test_folder_path = '/Users/annafabia/Desktop/NIDS_deep_images/attacks_classifier/malNOSMOTE_test_images'

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

model.save('NO_SMOTE_class_model.keras')

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')

# Make predictions on the test data
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_one_hot, axis=1)

# Calculate F1 score on test set
test_f1_score = f1_score(true_classes, predicted_classes, average='weighted')
print(f'Test F1 Score: {test_f1_score:.4f}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Extract values from the confusion matrix for each class
TP = np.diag(conf_matrix)  # True Positives
FP = conf_matrix.sum(axis=0) - TP  # False Positives
FN = conf_matrix.sum(axis=1) - TP  # False Negatives
TN = conf_matrix.sum() - (FP + FN + TP)  # True Negatives

# Calculate Precision, Recall, FPR, and FNR for each class
precision = TP / (TP + FP)
recall = TP / (TP + FN)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Print the values for each class
for i, class_label in enumerate(label_encoder.classes_):
    print(f"Class: {class_label}")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  FPR (False Positive Rate): {FPR[i]:.4f}")
    print(f"  FNR (False Negative Rate): {FNR[i]:.4f}")

# Print overall metrics for all classes (macro-average)
overall_precision = precision_score(true_classes, predicted_classes, average='weighted')
overall_recall = recall_score(true_classes, predicted_classes, average='weighted')

print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')


