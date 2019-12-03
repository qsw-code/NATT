'''
@author: Hao Wu, ShaoWei Qin
'''
import tensorflow as tf

def attention(inputs,num_kernel):

    w_omega = tf.Variable(tf.random_normal([2*num_kernel, 100], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([100], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([100], stddev=0.1))

    input_tensor=inputs

    v = tf.tanh(tf.tensordot(input_tensor, w_omega, axes=1) + b_omega)
    vu = tf.tensordot(v,u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')         
    att_out = tf.reduce_sum(input_tensor * tf.expand_dims(alphas, -1), 1)

    return att_out