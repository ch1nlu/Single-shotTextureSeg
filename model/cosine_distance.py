from tensorflow.keras import backend as K

def cosine_distance(vests):
    x, y = vests
    # x = K.l2_normalize(x, axis=-1)
    # y = K.l2_normalize(y, axis=-1)
    return 1+K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)