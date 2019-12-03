'''
@author: Hao Wu, ShaoWei Qin
'''
import tensorflow as tf
from keras.layers import Bidirectional,CuDNNGRU,Dense,Dropout


def text_encode(inputs,num_kernel):

    w_omega = tf.Variable(tf.random_normal([2*num_kernel, 100], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([100], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([100], stddev=0.1))
    
    lstm_input = inputs
   
    bilstm = Bidirectional(CuDNNGRU(num_kernel, return_sequences=True))(lstm_input)
    
    v = tf.tanh(tf.tensordot(bilstm, w_omega, axes=1) + b_omega)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  
    alphas = tf.nn.softmax(vu, name='alphas')         
    output = tf.reduce_sum(bilstm * tf.expand_dims(alphas, -1), 1)
    
    return output