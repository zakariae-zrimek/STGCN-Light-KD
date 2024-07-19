import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D, Bidirectional, GRU,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from IPython.core.debugger import set_trace
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler


class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoach=200, batch_size=10,cp1="stgcn",cp2="lstm"):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.lr = lr
        self.epoach =epoach
        self.batch_size = batch_size
        self.num_joints = 33
        # self.num_joints = 33

        self.cp1 = cp1
        self.cp2 = cp2


    def sgcn_gru(self, Input):
            # Input (None, None, 25, 3)
        """Temporal convolution"""
        k1 = tf.keras.layers.Conv2D(64, (9,1), padding='same', activation='relu')(Input)
        # k1 = BatchNormalization()(k1)
        # k1 = Dropout(0.5)(k1)
        k =  concatenate([Input, k1], axis=-1) #shape=(None, None, 25, 67)
        """Graph Convolution"""

        """first hop localization"""
        x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
        # x1 = BatchNormalization()(x1)
        # x1 = Dropout(0.5)(x1)
        x1 = tf.keras.layers.Conv2D(filters=33, kernel_size=(1,1), strides=1, activation='relu')(x1)
        # x1 = BatchNormalization()(x1)
        # x1 = Dropout(0.5)(x1)
        # f_1 = self.transformer_encoder(x1,64,3,33,0)
        x_dim = tf.keras.layers.Reshape(target_shape=(-1,x1.shape[2]*x1.shape[3]))(x1) # (None, None, 1200)
        f_1 = GRU(33, return_sequences=True)(x_dim) #(10,100,33)
        f_1 = tf.expand_dims(f_1, axis=3)
        logits = f_1 
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1]) #(None, None, 33, 64)

        """second hop localization"""
        y1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), strides=1, activation='relu')(k)
        # y1 = Dropout(0.5)(y1)
        # y1 = BatchNormalization()(y1)
        y1 = tf.keras.layers.Conv2D(filters=33, kernel_size=(1,1), strides=1, activation='relu')(y1)
        # y1 = Dropout(0.5)(y1)
        # y1 = BatchNormalization()(y1)
        # f_2 = TransformerEncoder(intermediate_dim=64, num_heads=3)(y1)
        # f_2 = self.transformer_encoder(y1,64,3,33,0)
        y_dim = tf.keras.layers.Reshape(target_shape=(-1,x1.shape[2]*x1.shape[3]))(y1)
        f_2 = GRU(33, return_sequences=True)(y_dim)
        f_2 = tf.expand_dims(f_2, axis=3)
        logits = f_2   
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1]) #(None, None, 33, 64)
        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)  #(None, None, 33, 128)
        
        """Temporal convolution"""
        z1 = tf.keras.layers.Conv2D(16, (9,1), padding='same', activation='relu')(gcn_1)
        # z1 = Dropout(0.5)(z1)
        # z1 = BatchNormalization()(z1)
        z2 = tf.keras.layers.Conv2D(16, (15,1), padding='same', activation='relu')(z1)
        # z2 = BatchNormalization()(z2)
        # z2 = Dropout(0.5)(z2) #(None, None, 25, 16)   f_2
        z3 = tf.keras.layers.Conv2D(16, (20,1), padding='same', activation='relu')(z2)
        # z3 = BatchNormalization()(z3)
        # z3 = Dropout(0.5)(z3)  #(None, None, 25, 16)  f_3
        z = concatenate([z1, z2, z3], axis=-1) #(None, None, 25, 48)   => Y = f_1 concat f_2  concat f_3
        
        return z


    def Lstm(self,x):
        x = tf.keras.layers.Reshape(target_shape=(-1,x.shape[2]*x.shape[3]))(x)
        rec = LSTM(60, return_sequences=True)(x)
        rec = Dropout(0.2)(rec)
        rec1 = LSTM(20, return_sequences=True)(rec)
        rec1 = Dropout(0.2)(rec1)
        rec2 = LSTM(20, return_sequences=True)(rec1)
        rec2 = Dropout(0.2)(rec2)
        rec3 = LSTM(60)(rec2)
        rec3 = Dropout(0.2)(rec3)
        rec3 = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.05))(rec3)
        rec3 = Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.05))(rec3)
        out = Dense(1, activation='sigmoid')(rec3)
        return out

  

    def train(self):
        seq_input = Input(shape=(70, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        
        if self.cp1 =="stgcn": 
            print("======================== CP1 GRU ===========================")
            x = self.sgcn_gru(seq_input)
            y = self.sgcn_gru(x)
            y = y + x
            z = self.sgcn_gru(y)
            z = z + y   

        if self.cp2 =="lstm": 
            print("======================== CP2 LSTM ===========================")
            out = self.Lstm(z)
           
        # weights = {0:2 , 1:1}
        self.model = Model(seq_input, out)
        initial_learning_rate = self.lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )

        # Compile the model with the learning rate scheduler
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        )

        checkpoint = ModelCheckpoint("teacher_model/best_model.hdf5", monitor='val_loss', save_best_only=True, mode='auto', period=1)
        # history = self.model.fit(self.train_x, self.train_y, validation_split=0.2, epochs=self.epoach, shuffle=True, batch_size=self.batch_size,class_weight = weights, callbacks=[checkpoint])
        history = self.model.fit(self.train_x, self.train_y, validation_split=0.2, epochs=self.epoach, shuffle=True, batch_size=self.batch_size, callbacks=[checkpoint])
       
        # serialize model to json
        json_model = self.model.to_json()

        with open('best model ex4/rehabilitation.json', 'w') as json_file:
            json_file.write(json_model)
       
       
        return history
      
    def prediction(self, data):
        y_pred = self.model.predict(data)

        return y_pred
    