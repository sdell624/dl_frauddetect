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

    def call(self, inputs, training=False):
        # print(inputs.shape, " is inputs")
        inputs = tf.expand_dims(inputs, axis=-1) #expand for hidden_dim
        # print(inputs.shape, " is new inputs")
        
        x = self.embedding(inputs)
        x = self.conv1d(x)
        x = self.pooling(x)
        return self.classifier(x)
    
x_train, y_train, x_test, y_test = pre.anomaly_preprocess(train_percentage=0.8, train_balanced=True)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print number of fraudulent samples in training and test set
print(f'Fraudulent samples in training set: {tf.reduce_sum(y_train)}')
print(f'Fraudulent samples in test set: {tf.reduce_sum(y_test)}')

# Create model
# model = CNN1D(num_filters=64, kernel_size=3, hidden_dim=64, dropout_rate=0.2)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=32)
# loss, accuracy = model.evaluate(x_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')