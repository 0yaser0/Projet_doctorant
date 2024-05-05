# model_training.py
#This file contains functions for training different models.

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow import keras

def train_svm(X_train, y_train, X_test, y_test):
    # Reshape the data for SVM classifier
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # Create and train SVM classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train_flattened, y_train.ravel())

    # Make predictions
    y_pred = svm_classifier.predict(X_test_flattened)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("SVM Accuracy:", accuracy)
    return y_pred

def train_cnn(X_train, y_train, X_test, y_test):
    model = keras.models.Sequential([
            keras.Input(shape=(240, 240, 1)),
            # Add layers as per your CNN architecture
        ])
    # Compile the model
    # Train the model
    # Evaluate the model
    return history
