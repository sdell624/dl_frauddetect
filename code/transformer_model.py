import pandas as pd
import tensorflow as tf
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
        x = Input(shape=(6, self.hidden_dim))
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
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.pooling(x)
        return self.classifier(x)


# Import data
df = pd.read_csv('../data/card_transdata.csv')
df = df.iloc[:, 1:] #Drop header
features = df.iloc[:, :-1] 
labels = df.iloc[:, -1] 
data = features.values

#Split
train_size = int(0.8 * len(data))
X_train, y_train = data[:train_size], labels[:train_size]
X_test, y_test = data[train_size:], labels[train_size:]


#Conver to tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)
print("Tensor shape:", X_train.shape)
#print(labels)

# Example usage:
num_heads = 2
hidden_dim = 32
num_layers = 2
dropout_rate = 0.1

model = Transformer(num_heads, hidden_dim, num_layers, dropout_rate)
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()
accuracy_metric = BinaryAccuracy()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=[accuracy_metric])

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2) 
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
#predictions = model.predict(X_new_data)

