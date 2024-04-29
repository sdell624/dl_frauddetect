import pandas as pd
import numpy as np
import tensorflow as tf

# Anomaly Detection Dataset (real data, not encoded)
# https://www.kaggle.com/code/rudyhendraprasetiya/anomaly-detection-with-autoencoders/input
def anomaly_preprocess(train_percentage=0.8, train_balanced=False):
    # Import data and shuffle
    np.random.seed(0)
    df = pd.read_csv('../data/card_transdata.csv')
    df = df.sample(frac=1).reset_index(drop=True)

    # Split into train and test set
    train_size = int(train_percentage * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Balance the training data if desired
    if train_balanced:
        # Get all samples from training data that are fraudulent
        fraud_data = train_data[train_data['fraud'] == 1]
        # Sample the same number of non-fraudulent samples from the training data
        non_fraud_data = train_data[train_data['fraud'] == 0].sample(len(fraud_data))
        # Combine these to give a balanced training set (half fraud, half non-fraud)
        train_data = pd.concat([fraud_data, non_fraud_data])
        # Shuffle again, since right now all fraudulent samples are at the top
        train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Using .to_numpy() drops the column headers
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    
    # Split into features and labels
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    x_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Convert data to tensors and return
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)

    return x_train, y_train, x_test, y_test

def synthetic_preprocess(train_percentage=0.8, train_balanced=False):
    
    #Drop column
    
    #Balance
    
    #ohe
    
    #Split
    
    
    pass

# Sanity checks
# x_train, y_train, x_test, y_test = anomaly_preprocess(train_percentage=0.8, train_balanced=True)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(x_train[:5].numpy())
# print(y_train[:5].numpy())
# print(x_test[:5].numpy())
# print(y_test[:5].numpy())

# print(f'Fraudulent samples in training set: {np.sum(y_train.numpy())}')
# print(f'Fraudulent samples in test set: {np.sum(y_test.numpy())}')