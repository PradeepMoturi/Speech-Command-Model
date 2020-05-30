import os
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K

os.environ['PYTHONHASHSEED']='123'
np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)

data_dir = '../Data/Pradeep_16'

x_train = np.load(data_dir+'mfcc_train.npy')
y_train = np.load(data_dir+'y_train.npy')
x_test = np.load(data_dir+'mfcc_test.npy')
y_test = np.load(data_dir+'y_test.npy')

def AttentionModel(sr=16000, iLen=25000):
    
    inputs = L.Input(x_train.shape[1:], name='input')

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(inputs)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x =  K.squeeze(x,axis=-1)

    n_units = 64
    x = L.LSTM(n_units, return_sequences=True)(x)  

    xLast = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(64)(xLast)

    # Calculate attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  

    x = L.Dot(axes=[1, 1])([attScores, x])  
    x = L.Dense(32, activation='relu')(x)
    output = L.Dense(5, activation='softmax', name='output')(x)
    model = Model(inputs=[inputs], outputs=[output])

    return model

model = AttentionModel()
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1,epochs=5,shuffle=True,batch_size=5)

model.save("model.h5")