import pandas as pd
import numpy as np
import tensorflow as tf

# Anomaly Detection Dataset (real data, not encoded)
# https://www.kaggle.com/code/rudyhendraprasetiya/anomaly-detection-with-autoencoders/input
def anomaly_preprocess(train_percentage=0.8, train_balanced=False, data_path='../data/card_transdata.csv', percent_fraud=0.5):
    # Import data and shuffle
    np.random.seed(0)
    df = pd.read_csv(data_path)
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
        total_samples = int(len(fraud_data) / percent_fraud)
        amount_non_fraud = total_samples - len(fraud_data)
        non_fraud_data = train_data[train_data['fraud'] == 0].sample(amount_non_fraud)
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

# Synthetically Generated Data (fake data, not encoded)
# https://www.kaggle.com/datasets/ealaxi/paysim1/data
def synthetic_preprocess(train_percentage=0.8, train_balanced=False, data_path='../data/synthetic_data.csv', percent_fraud=0.5):
    # Import data and shuffle
    np.random.seed(0)
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Split into train and test set
    train_size = int(train_percentage * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Balance the training data if desired
    if train_balanced:
        # Get all samples from training data that are fraudulent
        fraud_data = train_data[train_data['isFraud'] == 1]
        # Sample the same number of non-fraudulent samples from the training data
        total_samples = int(len(fraud_data) / percent_fraud)
        amount_non_fraud = total_samples - len(fraud_data)
        non_fraud_data = train_data[train_data['isFraud'] == 0].sample(amount_non_fraud)
        # Combine these to give a balanced training set (half fraud, half non-fraud)
        train_data = pd.concat([fraud_data, non_fraud_data])
        # Shuffle again, since right now all fraudulent samples are at the top
        train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Separate features and labels (column 'isFraud' is the label)
    y_train = train_data['isFraud']
    x_train = train_data.drop(columns=['isFraud'])
    y_test = test_data['isFraud']
    x_test = test_data.drop(columns=['isFraud'])
    
    # Drop categorical columns (type, nameOrig, nameDest)
    x_train = x_train.drop(columns=['type', 'nameOrig', 'nameDest'])
    x_test = x_test.drop(columns=['type', 'nameOrig', 'nameDest'])

    # Ideally would one-hot encode instead of just dropping these columns,
    #    but doing so for any significant portion of the datset uses too much memory
    # x_train = pd.get_dummies(x_train, columns=['type', 'nameOrig', 'nameDest'])
    # x_test = pd.get_dummies(x_test, columns=['type', 'nameOrig', 'nameDest'])

    # Use .to_numpy() to drop the column headers
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    # Convert data to tensors and return
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int8)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int8)

    return x_train, y_train, x_test, y_test

# Real PCA Data (real data, already encoded with PCA so features do not have direct meaning except for 'Time' and 'Amount')
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
def pca_preprocess(train_percentage=0.8, train_balanced=False, data_path='../data/pca_data.csv', percent_fraud=0.5):
    # Import data and shuffle
    np.random.seed(0)
    df = pd.read_csv(data_path)
    df = df.sample(frac=1).reset_index(drop=True)

    # Split into train and test set
    train_size = int(train_percentage * len(df))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    # Balance the training data if desired
    if train_balanced:
        # Get all samples from training data that are fraudulent
        fraud_data = train_data[train_data['Class'] == 1]
        # Sample the same number of non-fraudulent samples from the training data
        total_samples = int(len(fraud_data) / percent_fraud)
        amount_non_fraud = total_samples - len(fraud_data)
        non_fraud_data = train_data[train_data['Class'] == 0].sample(amount_non_fraud)
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

# Sanity checks
# x_train, y_train, x_test, y_test = pca_preprocess(train_percentage=0.8, train_balanced=False)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(x_train[:5].numpy())
# print(y_train[:5].numpy())
# print(x_test[:5].numpy())
# print(y_test[:5].numpy())

# print(f'Fraudulent samples in training set: {np.sum(y_train.numpy())}')
# print(f'Fraudulent samples in test set: {np.sum(y_test.numpy())}')