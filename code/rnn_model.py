import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.activations import gelu

class RNN(tf.keras.Model):
    def __init__(self, rnn_size=128):
        super().__init__()
        # Substitues for "embedding layer"
        self.d1 = tf.keras.layers.Dense(rnn_size)
        self.rnn = tf.keras.layers.LSTM(rnn_size, return_sequences=True, dropout=.1)
        self.d2 = tf.keras.layers.Dense(rnn_size, activation="relu")
        self.d3 = tf.keras.layers.Dense(32, activation="relu")
        self.d4 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.fixOutputs = tf.keras.layers.Reshape((-1, 1)) 

    def call(self, inputs):
        logits = self.rnn(inputs) 
        outputs = self.d4(self.d3(self.d2(self.d1(logits))))
        correctedOutputs = self.fixOutputs(outputs)
        return correctedOutputs


# #load data from preprocess
# df = pd.read_csv('data/card_transdata.csv')
# #87,403 fraud
# #174,806 total for 50% split
# #Drop 825194 non fraud for equal balance
# df = df.drop(df[(df['fraud'] == 0.0)].head(47403).index)
# # df = df.drop(df[(df['fraud'] == 0.0)].head(14806).index)
# # Shuffle
# df = df.sample(frac=1).reset_index(drop=True)



df = pd.read_csv('data/encoded_dataset.csv') #'Class' is last column
# print(df.head())
# print(len(df[(df['Class'] == 0)]), " non fraud")
# print(len(df[(df['Class'] == 1)]), " fraud")
#Drop 283823 non fraud for 50% fraud
# df = df.drop(df[(df['Class'] == 0)].head(283823).index)
# #Drop 28185 non fraud for 20% fraud
# df = df.drop(df[(df['Class'] == 0)].head(282347).index)


df = df.iloc[:, 1:] #Drop header
features = df.iloc[:, :-1] 
labels = df.iloc[:, -1] 
data = features.values

train_size = int(0.8 * len(data))
X_train, y_train = data[:train_size], labels[:train_size]
X_test, y_test = data[train_size:], labels[train_size:]

# # RNN expects recog
X_train = tf.reshape(tf.convert_to_tensor(X_train, dtype=tf.float32), (-1, 1, X_train.shape[1]))
Y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.reshape(tf.convert_to_tensor(X_test, dtype=tf.float32), (-1, 1, X_train.shape[1]))
Y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

model = RNN()
optimizer = Adam(learning_rate=0.005)
loss_fnn = BinaryCrossentropy()
accuracy_metric = BinaryAccuracy()
model.compile(optimizer=optimizer, loss=loss_fnn, metrics=[accuracy_metric])
model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_split = .2) 
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
predictions = model.predict(X_test)
