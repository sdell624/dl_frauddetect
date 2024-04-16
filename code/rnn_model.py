import tensorflow as tf
import numpy as np


#load data from preprocess
data = []



def __init__(self, rnn_size=128):    
    self.rnn = tf.keras.layers.SimpleRNN(rnn_size)
    self.dense = tf.keras.layers.Dense(1) 
    pass

def call(self, inputs):
   logits = self.rnn(inputs)    
   outputs = self.dense(logits)   
   return outputs


