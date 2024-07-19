import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, optimizers, losses, metrics
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, Conv2D, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D, Bidirectional, GRU,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from Graph_gru_lstm_model.graph_adjacency_processor import Graph

# Définition des constantes
REGULARIZER = tf.keras.regularizers.l2(l=0.0001)
INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2., mode="fan_out", distribution="truncated_normal")

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            ) * (self.temperature ** 2)

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict mapping metric names to current value
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, y = data

        # Forward pass
        y_prediction = self.student(x, training=False)

        # Updates the metrics tracking the loss
        student_loss = self.student_loss_fn(y, y_prediction)
        self.compiled_loss(y, y_prediction, regularization_losses=self.losses)

        # Update metrics
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict mapping metric names to current value
        return {"loss": student_loss, **{m.name: m.result() for m in self.metrics}}

    def call(self, x):
        return self.student(x)

class SimpleSgcnLstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.lr = lr
        self.num_joints = 33
        self.model = self.build_model()

    def sgcn_gru(self , Input):
        # Convolution layers
        k1 = tf.keras.layers.Conv2D(32, (9, 1), padding='same', activation='relu', kernel_initializer=INITIALIZER)(Input)
        k = concatenate([Input, k1], axis=-1)
        x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=1, activation='relu')(k)
        x1 = tf.keras.layers.Conv2D(filters=33, kernel_size=(1, 1), strides=1, activation='relu')(x1)

        # Reshape for GRU input
        x_dim = tf.keras.layers.Reshape(target_shape=(-1, x1.shape[2] * x1.shape[3]))(x1)

        # GRU layer
        f_1 = GRU(33, return_sequences=True)(x_dim)
        f_1 = tf.expand_dims(f_1, axis=3)

        # Logits and softmax
        logits = f_1
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)

        # Graph convolution with Lambda layer
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])

        # Second set of convolution layers
        y1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=1, activation='relu')(k)
        y1 = tf.keras.layers.Conv2D(filters=33, kernel_size=(1, 1), strides=1, activation='relu')(y1)
        y_dim = tf.keras.layers.Reshape(target_shape=(-1, x1.shape[2] * x1.shape[3]))(y1)
        f_2 = GRU(33, return_sequences=True)(y_dim)
        f_2 = tf.expand_dims(f_2, axis=3)
        logits = f_2
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1])

        # Concatenate and additional convolutions
        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)
        z1 = tf.keras.layers.Conv2D(16, (9, 1), padding='same', activation='relu')(gcn_1)
        z2 = tf.keras.layers.Conv2D(16, (15, 1), padding='same', activation='relu')(z1)
        z3 = tf.keras.layers.Conv2D(16, (20, 1), padding='same', activation='relu')(z2)
        z = concatenate([z1, z2, z3], axis=-1)

        return z

    def Lstm(self , x):
        # Reshape for LSTM input
        x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2] * x.shape[3]))(x)
        
        # LSTM layer with fewer units
        rec = LSTM(16, return_sequences=False)(x)  # Reduced number of units
        
        # Dense layer with BatchNormalization and Dropout
        rec = Dense(8, activation='relu')(rec)
        rec = BatchNormalization()(rec)  # Added BatchNormalization
        rec = Dropout(0.2)(rec)  # Dropout layer
        
        # Output layer for binary classification
        out = Dense(1, activation='sigmoid')(rec)
        
        return out

    def build_model(self):
        seq_input = Input(shape=(70, self.train_x.shape[2], self.train_x.shape[3]))
        x = self.sgcn_gru(seq_input)
        y = self.sgcn_gru(x)
        y = y + x
        z = self.sgcn_gru(y)
        z = z + y
        out = self.Lstm(z)
        model = Model(seq_input, out)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='binary_crossentropy')
        return model


def evaluate_model(model, test_x, test_y):
    results = model.evaluate(test_x, test_y)
    print(f"Test Loss: {results}")

# Chargement des données
x_train = np.load('Data_loader/train_x.npy')
y_train = np.load('Data_loader/train_y.npy')

# Séparation des données d'entraînement et de test
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, stratify=y_train, test_size=0.20, random_state=42)


# Initialisation du modèle
graph = Graph(33)
student_model_instance = SimpleSgcnLstm(train_x, train_y, graph.AD, graph.AD2, graph.bias_mat_1, graph.bias_mat_2, lr=0.0001)
student_model = student_model_instance.model

# Charger le modèle enseignant pré-entraîné
teacher_model_path = 'teacher_model/best_model.hdf5'
teacher_model = tf.keras.models.load_model(teacher_model_path, compile=False)


# Initialiser et compiler le distiller
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=optimizers.Adam(),
    metrics=[metrics.BinaryAccuracy()],
    student_loss_fn=losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)
"""Train the algorithm"""
import time
start = time.time()
# Distillation des connaissances du modèle enseignant au modèle étudiant avec validation_split
history = distiller.fit(train_x, train_y, epochs=50, batch_size=32, validation_split=0.2)
print("***************************** Temps d'excution*****************************")
print(time.time() - start)
# Sauvegarder le modèle étudiant
distiller.student.save("student_model/student_model.h5")
print("Student model saved successfully.")

# Charger le modèle étudiant sauvegardé
student_model = tf.keras.models.load_model("student_model/student_model.h5")
print("Student model loaded successfully.")

# Prédiction avec le modèle étudiant
predictions = student_model.predict(test_x)

# Convert predictions to binary class labels
y_pred = (predictions > 0.5).astype(int).flatten()
test_y = test_y.flatten()

# Évaluer le modèle étudiant
accuracy = accuracy_score(test_y, y_pred)
print("Test Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(test_y, y_pred))

# Affichage de la matrice de confusion
conf_matrix = confusion_matrix(test_y, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='g', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Vérifiez les clés disponibles dans history.history
print(history.history.keys())

# Tracer les courbes de perte si elles existent
if 'loss' in history.history and 'val_loss' in history.history:
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
else:
    print("Les clés 'loss' et 'val_loss' n'existent pas dans history.history")
