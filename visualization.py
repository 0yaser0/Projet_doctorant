# visualization.py
#This file contains functions for visualizing model performance.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    classes = ['TUMOR', 'NoTUMOR']
    tick_marks = [0.5, 1.5]
    cn = confusion_matrix(y_true, y_pred)
    sns.heatmap(cn, cmap='plasma', annot=True)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_training_history(history):
    # Plot training history (e.g., loss, accuracy)
    pass
