import tensorflow as tf
import pandas as pd

class CNN1D(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, hidden_dim, dropout_rate):
        super(CNN1D, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.embedding = tf.keras.layers.Dense(hidden_dim, activation='leaky_relu')
        self.conv1d = tf.keras.layers.Conv1D(num_filters, kernel_size, activation='leaky_relu')
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # print(inputs.shape, " is inputs")
        inputs = tf.expand_dims(inputs, axis=-1) #expand for hidden_dim
        # print(inputs.shape, " is new inputs")
        
        x = self.embedding(inputs)
        x = self.conv1d(x)
        x = self.pooling(x)
        return self.classifier(x)
    
# Import data (from https://www.kaggle.com/code/rudyhendraprasetiya/anomaly-detection-with-autoencoders/input)
df = pd.read_csv('../data/card_transdata.csv')
# Converting to numpy extracts values without headers
features = df.iloc[:, :-1].to_numpy()
labels = df.iloc[:, -1].to_numpy()

# Split into train and test
train_size = int(0.8 * len(features))
x_train, y_train = features[:train_size], labels[:train_size]
x_test, y_test = features[train_size:], labels[train_size:]

# Convert data to tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)

# Create model
model = CNN1D(num_filters=64, kernel_size=3, hidden_dim=64, dropout_rate=0.2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')