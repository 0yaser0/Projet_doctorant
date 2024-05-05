# main.py
# This is the main script that orchestrates the entire process.

from data_processing import load_data
from model_training import train_svm, train_cnn
from visualization import plot_confusion_matrix, plot_training_history
from sklearn.model_selection import train_test_split

# Load data
X, y = load_data(['path_to_yes_folder', 'path_to_no_folder'], (240, 240))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_pred = train_svm(X_train, y_train, X_test, y_test)

# Train CNN
cnn_history = train_cnn(X_train, y_train, X_test, y_test)

# Visualize performance
plot_confusion_matrix(y_test, svm_pred)
plot_training_history(cnn_history)
