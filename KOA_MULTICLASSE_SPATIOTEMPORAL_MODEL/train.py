import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from Graph_gru_lstm_model.data_prep_augmentation import Data_Loader
from Graph_gru_lstm_model.graph_adjacency_processor import Graph
from Graph_gru_lstm_model.gru_lstm_hybrid_network import Sgcn_Lstm
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import time
random_seed = 42  # for reproducibility

# Créer le parser
my_parser = argparse.ArgumentParser(description='Liste des arguments')

# Ajouter les arguments
my_parser.add_argument('--ex', type=str, default='Kimore_ex5',
                       help="le nom de l'exercice.", required=True)
my_parser.add_argument('--lr', type=float, default=0.004,
                       help="taux d'apprentissage initial pour l'optimiseur.")
my_parser.add_argument('--epoch', type=int, default=1000,
                       help="nombre d'époques pour l'entraînement.")
my_parser.add_argument('--batch_size', type=int, default=10,
                       help="taille de batch pour l'entraînement.")
my_parser.add_argument('--cp1', type=str, default='stgcn',
                       help="premier point de contrôle.")
my_parser.add_argument('--cp2', type=str, default='lstm',
                       help="deuxième point de contrôle.")

# Exécuter la méthode parse_args()
args = my_parser.parse_args()

"""Importer l'ensemble de données"""
X_train = Data_Loader('Data').scaled_x
y_train = Data_Loader('Data').scaled_y
"""Importer la structure de données du graphe"""
graph = Graph(33)

"""Diviser les données en ensembles d'entraînement et de validation tout en préservant la distribution"""
train_x, test_x, train_y, test_y = train_test_split(X_train, y_train, stratify=y_train, test_size=0.20, random_state=42)

print(train_x.shape, test_x.shape)

# ====================================================================================================================
# ====================================================================================================================
# ====================================================================================================================

"""Entraîner l'algorithme"""
start = time.time()

algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1,
                       graph.bias_mat_2, lr=args.lr, epoch=args.epoch,
                         batch_size=args.batch_size, cp1=args.cp1, cp2=args.cp2
                        )
history = algorithm.train()

print("***************************** Temps d'exécution *****************************")
print(time.time() - start)

"""Tester le modèle"""
y_pred = algorithm.prediction(test_x)

print(y_pred)
print(test_y.shape)
print(y_pred.shape)

y_pred = y_pred.argmax(axis=1)
print(test_y.shape)
print(y_pred.shape)

test_y = test_y.flatten()

y_pred = y_pred.flatten()
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

# Accuracy is sum of diagonal divided by total observations
cf = conf_matrix

stats_text = ""
accuracy = np.trace(cf) / float(np.sum(cf))

# if it is a binary confusion matrix, show some more stats
if len(cf) == 2:
    # Metrics for Binary Confusion Matrices
    precision = cf[1, 1] / sum(cf[:, 1])
    recall = cf[1, 1] / sum(cf[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f} - Precision={:0.3f} - Recall={:0.3f} - F1 Score={:0.3f}".format(
        accuracy, precision, recall, f1_score)
else:
    stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=['Early', 'Moderate', 'Severe'], yticklabels=['Early', 'Moderate', 'Severe'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels' + stats_text)
plt.ylabel('True Labels')
plt.show()

# Calculate and print classification report
print("Classification Report:")
print(classification_report(labels, predicted_labels_reshaped, digits=4))

# list all data in history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
