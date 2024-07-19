from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


random_seed = 42  # for reproducibility

# Charger les données
# data_loader = Data_Loader('Data')  # folder name -> Train.csv, Test.csv
train_x = np.load("Data_Loader2/train_x.npy")
train_y = np.load("Data_Loader2/train_y.npy")


train_x, test_x, train_y, test_y = train_test_split(
    train_x, train_y, stratify=train_y, test_size=0.20, random_state=random_seed
)
model = model = load_model("teacher_model/best_model.h5")
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

print(f"Moyenne F1 Score (macro): {f1_macro:.3f}")
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    xticklabels=["Early", "Moderate", "Severe"],
    yticklabels=["Early", "Moderate", "Severe"],
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels" + stats_text)
plt.ylabel("True Labels")
plt.show()

# Calculate and print classification report
print("Classification Report:")
print(classification_report(labels, predicted_labels_reshaped, digits=4))
