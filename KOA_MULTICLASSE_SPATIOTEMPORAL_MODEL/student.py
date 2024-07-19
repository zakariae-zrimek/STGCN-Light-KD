from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from Graph_gru_lstm_model.graph_adjacency_processor import Graph
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


random_seed = 42  # for reproducibility

# Charger les données
# data_loader = Data_Loader('Data')  # folder name -> Train.csv, Test.csv
train_x = np.load("Data_Loader2/train_x.npy")
train_y = np.load("Data_Loader2/train_y.npy")

print(train_x.shape)
train_x, test_x, train_y, test_y = train_test_split(
    train_x, train_y, stratify=train_y, test_size=0.20, random_state=random_seed
)
print('test-y', test_y.shape)
model = model = load_model("student_model/student_model.h5")
model.summary()
y_pred = model.predict(test_x)

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



sns.heatmap(conf_matrix, annot=True, fmt='g',xticklabels=['Early', 'Moderate','Severe'],yticklabels=['Early', 'Moderate','Severe'])
 
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels'+ stats_text)
plt.ylabel('True Labels')
plt.show()
 
# Calculate and print classification report
print("Classification Report:")
print(classification_report(labels, predicted_labels_reshaped, digits=4))
