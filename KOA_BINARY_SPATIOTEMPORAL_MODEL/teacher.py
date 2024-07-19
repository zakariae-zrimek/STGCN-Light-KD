import numpy as np
random_seed = 42  # for reproducibility
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
# from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import *

from sklearn.metrics import mean_squared_error, mean_absolute_error

import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import time




train_x = np.load('Data_loader/train_x.npy')
train_y = np.load('Data_loader/train_y.npy')


"""Split the data into training and validation sets while preserving the distribution"""
train_x, test_x, train_y, test_y = train_test_split(train_x,train_y,stratify=train_y, test_size=0.20, random_state = random_seed)

print(train_x.shape,test_x.shape)


#====================================================================================================================
#====================================================================================================================
#====================================================================================================================


"""Train the algorithm"""

start = time.time()

from tensorflow.keras.models import load_model

# Charger le modèle à partir du fichier HDF5
algorithm = load_model('teacher_model/best_model.hdf5')

algorithm.summary()
print("***************************** Temps d'excution*****************************")
print(time.time() - start)


# """Test the model"""
y_pred= algorithm.predict(test_x)



print(y_pred)
print(test_y.shape)
print(y_pred.shape)

y_pred = (y_pred > 0.5).astype(int)



print(test_y.shape)
print(y_pred.shape)



test_y = test_y.flatten()


y_pred =y_pred.flatten()
print(test_y)
print(y_pred)


accuracy = accuracy_score(test_y, y_pred)
print("Test Accuracy", accuracy)

print("Confusion Matrix", confusion_matrix(test_y, y_pred))


labels = test_y

predicted_labels_reshaped = y_pred

# Assuming predicted_labels_reshaped and labels are defined
conf_matrix = confusion_matrix(labels, predicted_labels_reshaped)
 
# Plotting the confusion matrix
# plt.figure(figsize=(6, 4))

print("###################################################")
print("###################################################")

#Accuracy is sum of diagonal divided by total observations
cf = conf_matrix

stats_text = ""
accuracy  = np.trace(cf) / float(np.sum(cf))
print("###################################################")
print("###################################################")

# Accuracy is sum of diagonal divided by total observations
cf = conf_matrix

stats_text = ""
accuracy = np.trace(cf) / float(np.sum(cf))

# Calcul de l'accuracy
accuracy = np.trace(cf) / np.sum(cf)

# Calcul de la sensitivity (rappel) pour chaque classe
sensitivity = np.zeros(cf.shape[0])
for i in range(cf.shape[0]):
    sensitivity[i] = cf[i, i] / np.sum(cf[i, :])

# Calcul de la specificity pour chaque classe
specificity = np.zeros(cf.shape[0])
for i in range(cf.shape[0]):
    true_negatives = np.sum(np.delete(np.delete(cf, i, axis=0), i, axis=1))
    all_negatives = np.sum(np.delete(cf, i, axis=1))
    specificity[i] = true_negatives / all_negatives

# Calcul de la precision pour chaque classe
precision = np.zeros(cf.shape[0])
for i in range(cf.shape[0]):
    precision[i] = cf[i, i] / np.sum(cf[:, i])

# Calcul du F1-score pour chaque classe
f1_score = np.zeros(cf.shape[0])
for i in range(cf.shape[0]):
    precision_i = precision[i]
    recall_i = sensitivity[i]
    f1_score[i] = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0

# Affichage des résultats par classe
for i in range(cf.shape[0]):
    print(f"Classe {i+1}:")
    print(f"  Sensitivity (Recall): {sensitivity[i]:.3f}")
    print(f"  Specificity: {specificity[i]:.3f}")
    print(f"  Precision: {precision[i]:.3f}")
    print(f"  F1 Score: {f1_score[i]:.3f}")
    print()

# Calcul de moyennes ou pondérations si nécessaire
# Exemple de moyenne macro pour le F1-score
f1_macro = np.mean(f1_score)

#  calcul de MCC
# Extraction des valeurs de la matrice de confusion
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

# Calcul du MCC à partir des valeurs extraites
def matthews_corr_coefficient(TP, TN, FP, FN):
    numerator = (TP * TN) - (FP * FN)
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    
    if denominator == 0:
        return 0  # Handle the case where denominator is zero (to avoid division by zero)
    
    mcc = numerator / denominator
    return mcc

# Calcul du MCC
mcc = matthews_corr_coefficient(TP, TN, FP, FN)
print("Matthews Correlation Coefficient (MCC):", mcc)

#if it is a binary confusion matrix, show some more stats
if len(cf)==2:
    #Metrics for Binary Confusion Matrices
    precision = cf[1,1] / sum(cf[:,1])
    recall    = cf[1,1] / sum(cf[1,:])
    f1_score  = 2*precision*recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f} - Precision={:0.3f} - Recall={:0.3f} - F1 Score={:0.3f}".format(
        accuracy,precision,recall,f1_score)
else:
    stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)



sns.heatmap(conf_matrix, annot=True, fmt='g',xticklabels=['NM', 'KOA'],yticklabels=['NM', 'KOA'])
 
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels'+ stats_text)
plt.ylabel('True Labels')
plt.show()
 
