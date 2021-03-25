import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model

from utils.model.testCorr import pairwise_cosine_sim


def new_model():
    query_input = Input(shape=(256, 256, 3))  # query image shape
    ref_input = Input(shape=(64, 64, 3))  # reference image shape

    # encoding
    query_embeddings, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 = encoding(query_input, 'query')
    ref_embeddings, _, _, _, _, _ = encoding(ref_input, 'ref')

    # correlation
    # corr = tf.einsum("ae,pe->ap", query_embeddings, ref_embeddings)  # must in rank 2
    # corr = keras.layers.Lambda(tf.nn.convolution)(query_embeddings, ref_embeddings, padding='SAME')
    corr = tf.nn.convolution(query_embeddings, ref_embeddings, padding='SAME')
    corr = tf.math.reduce_sum(corr, axis=-1, keepdims=True)
    # corr = keras.layers.Lambda(tf.nn.convolution, output_shape=(256, 256, 1),
    #                            arguments={'filter': ref_embeddings, 'padding': 'SAME'})(query_embeddings)
    # decoding
    prediction = decoding(corr, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    #
    model = Model(inputs=[query_input, ref_input], outputs=prediction)

    return model


def encoding(img_input, input_type):
    # VGG_Block 1
    conv1_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='conv1_1_' + input_type, kernel_initializer='he_normal')(img_input)
    conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='conv1_2_' + input_type, kernel_initializer='he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool' + input_type)(conv1_2)

    # VGG_Block 2
    conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='conv2_1_' + input_type, kernel_initializer='he_normal')(pool1)
    conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='conv2_2_' + input_type, kernel_initializer='he_normal')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool' + input_type)(conv2_2)

    # VGG_Block 3
    conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                     name='con3_1_' + input_type, kernel_initializer='he_normal')(pool2)
    conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                     name='conv3_2_' + input_type, kernel_initializer='he_normal')(conv3_1)
    conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                     name='conv3_3_' + input_type, kernel_initializer='he_normal')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool' + input_type)(conv3_3)

    # VGG_Block 4
    conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                     name='conv4_1_' + input_type, kernel_initializer='he_normal')(pool3)
    conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                     name='conv4_2_' + input_type, kernel_initializer='he_normal')(conv4_1)
    conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                     name='conv4_3_' + input_type, kernel_initializer='he_normal')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool' + input_type)(conv4_3)

    # VGG_Block 5
    conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                     name='conv5_1_' + input_type, kernel_initializer='he_normal')(pool4)
    # conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
    #                  name='conv5_2' + input_type, kernel_initializer='he_normal')(conv5_1)
    # conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
    #                  name='conv5_3' + input_type, kernel_initializer='he_normal')(conv5_2)
    # pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool' + input_type)(conv5_3)

    # Encoding
    res6_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res6_1_' + input_type, kernel_initializer='he_normal')(conv5_1)
    res6_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res6_2_' + input_type, kernel_initializer='he_normal')(res6_1)
    res6_3 = Conv2D(filters=512, kernel_size=3, activation=None, padding='same',
                    name='res6_3_' + input_type, kernel_initializer='he_normal')(res6_2)
    up6 = UpSampling2D(size=(2, 2))(res6_3)
    concat6 = concatenate([conv4_1, up6], axis=3)  # 32*32*1024

    res7_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res7_1_' + input_type, kernel_initializer='he_normal')(concat6)
    res7_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res7_2_' + input_type, kernel_initializer='he_normal')(res7_1)
    res7_3 = Conv2D(filters=512, kernel_size=3, activation=None, padding='same',
                    name='res7_3_' + input_type, kernel_initializer='he_normal')(res7_2)
    up7 = UpSampling2D(size=(2, 2))(res7_3)
    concat7 = concatenate([conv3_1, up7], axis=3)  # 64*64*768

    res8_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                    name='res8_1_' + input_type, kernel_initializer='he_normal')(concat7)
    res8_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                    name='res8_2_' + input_type, kernel_initializer='he_normal')(res8_1)
    res8_3 = Conv2D(filters=256, kernel_size=3, activation=None, padding='same',
                    name='res8_3_' + input_type, kernel_initializer='he_normal')(res8_2)
    up8 = UpSampling2D(size=(2, 2))(res8_3)
    concat8 = concatenate([conv2_1, up8], axis=3)  # 128*128*384

    res9_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                    name='res9_1_' + input_type, kernel_initializer='he_normal')(concat8)
    res9_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                    name='res9_2_' + input_type, kernel_initializer='he_normal')(res9_1)
    res9_3 = Conv2D(filters=128, kernel_size=3, activation=None, padding='same',
                    name='res9_3_' + input_type, kernel_initializer='he_normal')(res9_2)
    up9 = UpSampling2D(size=(2, 2))(res9_3)
    concat9 = concatenate([conv1_1, up9], axis=3)  # 256*256*192

    res10_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='res10_1_' + input_type, kernel_initializer='he_normal')(concat9)
    res10_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='res10_2_' + input_type, kernel_initializer='he_normal')(res10_1)
    res10_3 = Conv2D(filters=128, kernel_size=3, activation=None, padding='same',
                     name='res10_3_' + input_type, kernel_initializer='he_normal')(res10_2)
    encode_output = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                           name='encode_output' + input_type, kernel_initializer='he_normal')(res10_3)  # 256*256*64

    embeddings = tf.nn.l2_normalize(encode_output, axis=-1)

    return embeddings, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1


def decoding(corr, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1):
    pool11 = MaxPooling2D(pool_size=(16, 16), strides=(16, 16), name='pool11')(corr)  # 16*16*1
    conv11 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv11', kernel_initializer='he_normal')(conv5_1)  # 16*16*64 VGG features

    concat11 = concatenate([pool11, conv11], axis=3)  # 16*16*65
    res11_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res11_1', kernel_initializer='he_normal')(concat11)
    res11_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res11_2', kernel_initializer='he_normal')(res11_1)
    res11_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res11_3', kernel_initializer='he_normal')(res11_2)  # 16*16*64

    up12 = UpSampling2D(size=(2, 2))(res11_3)  # 32*32*64
    conv12 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv12', kernel_initializer='he_normal')(conv4_1)  # 32*32*64 VGG features
    pool12 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), name='pool12')(corr)  # 32*32*1
    concat12 = concatenate([up12, conv12, pool12], axis=3)  # 32*32*129
    res12_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res12_1', kernel_initializer='he_normal')(concat12)
    res12_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res12_2', kernel_initializer='he_normal')(res12_1)
    res12_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res12_3', kernel_initializer='he_normal')(res12_2)  # 32*32*64

    up13 = UpSampling2D(size=(2, 2))(res12_3)  # 64*64*64
    conv13 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv13', kernel_initializer='he_normal')(conv3_1)  # 64*64*64 VGG features
    pool13 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool13')(corr)  # 64*64*1
    concat13 = concatenate([up13, conv13, pool13], axis=3)  # 64*64*129
    res13_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res13_1', kernel_initializer='he_normal')(concat13)
    res13_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res13_2', kernel_initializer='he_normal')(res13_1)
    res13_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res13_3', kernel_initializer='he_normal')(res13_2)  # 64*64*64

    up14 = UpSampling2D(size=(2, 2))(res13_3)  # 128*128*64
    conv14 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv14', kernel_initializer='he_normal')(conv2_1)  # 128*128*64 VGG features
    pool14 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool14')(corr)  # 128*128*1
    concat14 = concatenate([up14, conv14, pool14], axis=3)  # 128*128*129
    res14_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res14_1', kernel_initializer='he_normal')(concat14)
    res14_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res14_2', kernel_initializer='he_normal')(res14_1)
    res14_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res14_3', kernel_initializer='he_normal')(res14_2)  # 128*128*64

    up15 = UpSampling2D(size=(2, 2))(res14_3)  # 256*256*64
    conv15 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv15', kernel_initializer='he_normal')(conv1_1)  # 256*256*64 VGG features 已经是64*64，应该可以省去
    # pool15 = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), name='pool15')(corr_input) # 128*128*1
    concat15 = concatenate([up15, conv15, corr], axis=3)  # 256*256*129
    res15_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res15_1', kernel_initializer='he_normal')(concat15)
    res15_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res15_2', kernel_initializer='he_normal')(res15_1)
    res15_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res15_3', kernel_initializer='he_normal')(res15_2)  # 256*256*64
    prediction = Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding='same',
                        name='prediction', kernel_initializer='he_normal')(res15_3)  # 256*256*1
    return prediction



keras.backend.clear_session()
strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = new_model()

    model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    optimizer = Adam(lr=0.00005)  # Learning rate = 0.00005
    model.compile(loss='binary_crossentropy', optimizer=optimizer, run_eagerly=True)

callbacks = [
    keras.callbacks.ModelCheckpoint("result.h5", save_best_only=True)
]
gt = np.load('D:\projects\OTS_reconstruct\\test_data\\test_ground_truth.npy')
# gt = gt[np.newaxis, :]
gt = np.expand_dims(gt, axis=-1)
query = np.load('D:\projects\OTS_reconstruct\\test_data\\test_query.npy')
query = query[np.newaxis, :]
ref = np.load('D:\projects\OTS_reconstruct\\test_data\\test_ref.npy')
ref = ref[np.newaxis, :]
history = model.fit(x=[query, ref], y=gt, batch_size=1, epochs=3, callbacks=callbacks)
