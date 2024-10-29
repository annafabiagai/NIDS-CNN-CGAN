'''
import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
#from keras.backend import clear_session, reset_uids

#--------------------------- LOADING IMAGES ------------------------------------

# Define function to load images and labels from directories
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Ensure grayscale
        img_array = np.array(img)
        images.append(img_array)
        
        # Extract label from the filename
        label = int(filename.split('_')[-1].split('.')[0])
        labels.append(label)
        
    return np.array(images), np.array(labels)

# Load the training and testing images
X_train, y_train = load_images_from_folder('raw_train_images')
X_test, y_test = load_images_from_folder('raw_test_images')

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to add channel dimension
X_train = X_train.reshape(-1, 7, 10, 1)
X_test = X_test.reshape(-1, 7, 10, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)



#------------------------ CNN MODEL -------------------------------
#48, 96

# Define the CNN model
def create_cnn_model():
    #clear_session()
    #reset_uids()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(7, 10, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))  # Output layer with softmax for categorical crossentropy
    return model

# Create and compile the CNN model
model = create_cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

model.save('binary_model.keras')



#------------------------------- EVALUATION OF THE MODEL --------------------------------


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Calculate evaluation metrics for class 1 (positive class)
FP = conf_matrix.sum(axis=0)[1] - np.diag(conf_matrix)[1]
FN = conf_matrix.sum(axis=1)[1] - np.diag(conf_matrix)[1]
TP = np.diag(conf_matrix)[1]
TN = conf_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)

# Precision or positive predictive value
PPV = TP / (TP + FP)

# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)

# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)

# Print evaluation metrics for class 1 (positive class)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'\nSensitivity (True Positive Rate): {TPR}')

print(f'Precision (Positive Predictive Value): {PPV}')

print(f'False Positive Rate: {FPR}')
print(f'False Negative Rate: {FNR}')

print(f'Overall Accuracy: {ACC}')

'''


#2 CONVOLUTIONAL LAYER CNN ---> ACHIEVED BETTER ACCURACY 

import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from keras.backend import clear_session, reset_uids
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#--------------------------- LOADING IMAGES ------------------------------------

# Define function to load images and labels from directories
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Ensure grayscale
        img_array = np.array(img)
        images.append(img_array)

        # Extract label from the filename
        label = int(filename.split('_')[-1].split('.')[0])
        labels.append(label)

    return np.array(images), np.array(labels)

# Load the training and testing images
X_train, y_train = load_images_from_folder('/Users/annafabia/Desktop/NIDS_deep_images/raw/raw_train_images')
X_test, y_test = load_images_from_folder('/Users/annafabia/Desktop/NIDS_deep_images/raw/raw_test_images')

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images to add channel dimension
X_train = X_train.reshape(-1, 7, 10, 1)
X_test = X_test.reshape(-1, 7, 10, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

#------------------------ CNN MODEL -------------------------------

# Define the CNN model
def create_cnn_model():
    #clear_session()
    #reset_uids()
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(7, 10, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))  # Output layer with softmax for categorical crossentropy
    return model

# Create and compile the CNN model
model = create_cnn_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

model.save('modified_binary_model.keras')

#------------------------------- EVALUATION OF THE MODEL --------------------------------

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Calculate evaluation metrics for class 1 (positive class)
FP = conf_matrix.sum(axis=0)[1] - np.diag(conf_matrix)[1]
FN = conf_matrix.sum(axis=1)[1] - np.diag(conf_matrix)[1]
TP = np.diag(conf_matrix)[1]
TN = conf_matrix.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)

# Precision or positive predictive value
PPV = TP / (TP + FP)

# Fall out or false positive rate
FPR = FP / (FP + TN)
# False negative rate
FNR = FN / (TP + FN)

# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)

# F1 Score
F1 = 2 * (PPV * TPR) / (PPV + TPR)

# Print evaluation metrics for class 1 (positive class)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'\nSensitivity (True Positive Rate): {TPR}')
print(f'Precision (Positive Predictive Value): {PPV}')
print(f'False Positive Rate: {FPR}')
print(f'False Negative Rate: {FNR}')
print(f'Overall Accuracy: {ACC}')
print(f'F1 Score: {F1}')



