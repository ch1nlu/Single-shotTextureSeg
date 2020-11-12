import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K

# set up test data
n_batch = 100
n = 400 # number of points in the first set
m = 500 # number of points in the second set
d = 200 # number of dimensions

A = np.random.rand(n_batch, n, d)
B = np.random.rand(n_batch, m, d)

# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm

def pairwise_cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A_tensor, K.permute_dimensions(B_tensor, (0,2,1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat =  num / den

    return dist_mat

# build dummy model
A_tensor = K.constant(A)
B_tensor = K.constant(B)
A_input = keras.Input(tensor=A_tensor)
B_input = keras.Input(tensor=B_tensor)
dist_output = keras.layers.Lambda(pairwise_cosine_sim)([A_input, B_input])
dist_model = keras.Model(inputs=[A_input, B_input], outputs=dist_output)
dist_model.compile("sgd", "mse")