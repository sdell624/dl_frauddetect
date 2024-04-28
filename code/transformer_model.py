import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.activations import gelu

class Transformer(tf.keras.Model):
    def __init__(self, num_heads, hidden_dim, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.embedding = Dense(hidden_dim, activation='leaky_relu')
        self.transformer_blocks = [self._build_transformer_block() for _ in range(num_layers)]
        self.pooling = GlobalAveragePooling1D()
        self.classifier = Dense(1, activation='sigmoid')

    def _build_transformer_block(self):
        x = Input(shape=(6, self.hidden_dim))   #6 is feature size 
        layer_norm = LayerNormalization(epsilon=1e-6)(x)
        dropout = Dropout(self.dropout_rate)(layer_norm)
        attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim)(dropout, dropout)
        dropout_2 = Dropout(self.dropout_rate)(attention)
        layer_norm_2 = LayerNormalization(epsilon=1e-6)(dropout_2)
        gelu_layer = gelu(layer_norm_2)
        dense_layer = Dense(self.hidden_dim, activation='leaky_relu')(gelu_layer)
        dropout_3 = Dropout(self.dropout_rate)(dense_layer)
        outputs = dense_layer + Dense(self.hidden_dim)(dropout_3)
        return Model(inputs=x, outputs=outputs)

    def call(self, inputs, training=False):
        print(inputs.shape, " is inputs")
        inputs = tf.expand_dims(inputs, axis=-1) #expand for hidden_dim
        inputs = tf.tile(inputs, [1, 1, self.hidden_dim])  #extend for hidden_dim
        print(inputs.shape, " is new inputs")
    
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.pooling(x)
        return self.classifier(x)


# Import data
df = pd.read_csv('../data/card_transdata.csv')
#87,403 fraud
#174,806 total for 50% split

#Drop 825194 non fraud for equal balance
df = df.drop(df[(df['fraud'] == 0.0)].head(825194).index)

#Shuffle
df = df.sample(frac=1).reset_index(drop=True)

df = df.iloc[:, 1:] #Drop header
features = df.iloc[:, :-1] 
labels = df.iloc[:, -1] 
data = features.values



#label_counts = df['fraud'].value_counts()
#print(label_counts[1], " is nonfraud")


#Split
train_size = int(0.8 * len(data))
X_train, y_train = data[:train_size], labels[:train_size]
X_test, y_test = data[train_size:], labels[train_size:]



#Conver to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)


# Example usage:
num_heads = 2
hidden_dim = 16
num_layers = 2
dropout_rate = 0.1

model = Transformer(num_heads, hidden_dim, num_layers, dropout_rate)
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()
accuracy_metric = BinaryAccuracy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])
#model.fit(X_train, y_train, batch_size=64, epochs=1, validation_split=0.2) 
#loss, accuracy = model.evaluate(X_test, y_test)
#print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#predictions = model.predict(X_new_data)

