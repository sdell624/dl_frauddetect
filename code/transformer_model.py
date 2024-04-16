import tensorflow as tf, tfm
import numpy as np



#load data from preprocess
data = []

def __init__(self, internediate_size=128):    
    self.rnn = tfm.nlp.layers.Transformer(3, internediate_size, 'leaky_relu')
    self.dense = tf.keras.layers.Dense(1) 
    pass

def call(self, inputs):
   logits = self.rnn(inputs)    
   outputs = self.dense(logits)   
   return outputs
