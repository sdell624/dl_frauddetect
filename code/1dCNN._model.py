import tensorflow as tf
import pandas as pd
import preprocess as pre

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

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1) #expand for hidden_dim
        x = self.embedding(inputs)
        x = self.conv1d(x)
        x = self.pooling(x)
        return self.classifier(x)
    
x_train, y_train, x_test, y_test = pre.anomaly_preprocess(train_percentage=0.8, train_balanced=False, percent_fraud=0.2)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Create and train model
model = CNN1D(num_filters=64, kernel_size=3, hidden_dim=64, dropout_rate=0.2)
loss = tf.keras.losses.BinaryCrossentropy()
accuracy = tf.keras.metrics.BinaryAccuracy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
print(model.summary())

# Calculate desired metrics (loss, accuracy, false negatives, false positives)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
fn = tf.keras.metrics.FalseNegatives()
fp = tf.keras.metrics.FalsePositives()
fn.update_state(y_test, y_pred)
fp.update_state(y_test, y_pred)
print(f'False Negatives: {fn.result().numpy()}, False Positives: {fp.result().numpy()}')