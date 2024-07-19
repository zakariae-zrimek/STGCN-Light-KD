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
from Graph_gru_lstm_model.data_prep_augmentation import Data_Loader
from Graph_gru_lstm_model.graph_adjacency_processor import Graph
from Graph_gru_lstm_model.gru_lstm_hybrid_network import Sgcn_Lstm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
import time
# Create the parser
my_parser = argparse.ArgumentParser(description='List of argument')

# Add the arguments
my_parser.add_argument('--ex', type=str, default='Kimore_ex5',
                       help='the name of exercise.', required=True)

my_parser.add_argument('--lr', type=int, default= 0.004,
                       help='initial learning rate for optimizer.')

my_parser.add_argument('--epoch', type=int, default= 1000,
                       help='number of epochs to train.')

my_parser.add_argument('--batch_size', type=int, default= 10,
                       help='training batch size.')


my_parser.add_argument('--cp1', type=str, default= 'stgcn',
                       help='initial learning rate for optimizer.')
my_parser.add_argument('--cp2', type=str, default= 'lstm',
                       help='initial learning rate for optimizer.')


#my_parser.add_argument('Path',
#                       type=str,
#                       help='the path to list')

# Execute the parse_args() method
args = my_parser.parse_args()






"""import the whole dataset"""
data_loader = Data_Loader(args.ex)  # folder name -> Train.csv, Test.csv

"""import the graph data structure"""
# graph = Graph(len(data_loader.body_part))

graph = Graph(33)

"""Split the data into training and validation sets while preserving the distribution"""
train_x, test_x, train_y, test_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y,stratify=data_loader.scaled_y, test_size=0.20, random_state = random_seed)

print(train_x.shape,test_x.shape)


#====================================================================================================================
#====================================================================================================================
#====================================================================================================================


"""Train the algorithm"""

start = time.time()


algorithm = Sgcn_Lstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr = args.lr, epoach=args.epoch, batch_size=args.batch_size,cp1=args.cp1,cp2=args.cp2)
history = algorithm.train()

print("***************************** Temps d'excution*****************************")
print(time.time() - start)


# """Test the model"""
y_pred= algorithm.prediction(test_x)




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



sns.heatmap(conf_matrix, annot=True, fmt='g',xticklabels=['Healthy', 'Unhealthy'],yticklabels=['Healthy', 'Unhealthy'])
 
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels'+ stats_text)
plt.ylabel('True Labels')
plt.show()
 

# list all data in history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

