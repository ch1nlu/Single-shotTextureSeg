import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt

# model using the pretrained VGG16 weights
from utils.model.correlation import spatially_convolve
from utils.model.loss import pixelwise_crossentropy
from utils.model.cosine_distance import cosine_distance


def new_model():
    vgg_query = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    vgg_query.trainable = True
    vgg_ref = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_ref.trainable = True
    for i in range(18):
        vgg_ref.layers[i]._name = vgg_ref.layers[i].name + '_ref'
        vgg_query.layers[i]._name = vgg_query.layers[i].name + '_query'
    ixs = [1, 4, 7, 11, 15]
    vgg_outputs_query = [vgg_query.layers[i].output for i in ixs]
    vgg_outputs_ref = [vgg_ref.layers[i].output for i in ixs]

    query_embeddings = encode(vgg_outputs_query, 'query')
    ref_embeddings = encode(vgg_outputs_ref, 'ref')

    # downsample 256 to 128
    query_embeddings = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='downsample_query_embedding')(
        query_embeddings)
    # upsample 64 to 128
    ref_embeddings = UpSampling2D(size=(2, 2), name='upsample_ref_embeddings')(ref_embeddings)
    # cosine dist
    corr = Lambda(cosine_distance, output_shape=(128, 128, 1), name='correlation')([query_embeddings, ref_embeddings])
    # upsample 128 to 256
    corr = UpSampling2D(size=(2, 2))(corr)  # (256, 256, 1)

    # Correlation
    # corr = tf.nn.convolution(query_embeddings, ref_embeddings, padding='SAME')
    # corr = tf.math.reduce_sum(corr, axis=-1, keepdims=True)
    # corr_layer = keras.layers.Lambda(spatially_convolve, output_shape=(256, 256, 1), name='correlation')
    # corr = corr_layer([query_embeddings, ref_embeddings])
    # corr = spatially_convolve([query_embeddings, ref_embeddings])
    # query_embeddings = tf.transpose(query_embeddings, [3, 0, 1, 2])
    # ref_embeddings = tf.transpose(ref_embeddings, [3, 0, 1, 2])
    # corr = tf.map_fn(lambda channel: tf.nn.convolution(input=query_embeddings, filters=ref_embeddings,
    #                                                    padding='SAME'))
    # corr = tf.nn.depthwise_conv2d(input=query_embeddings, filter=ref_embeddings, strides=[1, 1, 1, 1],
    #                               padding='SAME')
    # corr = tf.math.reduce_sum(corr, axis=-1, keepdims=True)
    # Decoding
    prediction = decoding(corr, vgg_outputs_query)

    model = Model(inputs=[vgg_query.inputs, vgg_ref.inputs], outputs=prediction)
    return model


def encode(inputs, input_type):
    # Res block
    conv6_1 = Conv2D(filters=512, kernel_size=1, activation='relu', padding='same')(inputs[4])  # 64*64*512
    res6_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res6_1_' + input_type, kernel_initializer='he_normal')(conv6_1)
    norm6_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res6_1)
    res6_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res6_2_' + input_type, kernel_initializer='he_normal')(norm6_1)
    norm6_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res6_2)
    res6_3 = Conv2D(filters=512, kernel_size=3, activation=None, padding='same',
                    name='res6_3_' + input_type, kernel_initializer='he_normal')(norm6_2)
    norm6_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res6_3)
    add6_3 = add([norm6_3, conv6_1])
    up6 = UpSampling2D(size=(2, 2))(add6_3)
    concat6 = concatenate([inputs[3], up6], axis=3)  # 32*32*1024

    conv7_1 = Conv2D(filters=512, kernel_size=1, activation='relu', padding='same')(concat6)  # 1*1 conv
    res7_1 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res7_1_' + input_type, kernel_initializer='he_normal')(conv7_1)
    norm7_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res7_1)
    res7_2 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same',
                    name='res7_2_' + input_type, kernel_initializer='he_normal')(norm7_1)
    norm7_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res7_2)
    res7_3 = Conv2D(filters=512, kernel_size=3, activation=None, padding='same',
                    name='res7_3_' + input_type, kernel_initializer='he_normal')(norm7_2)
    norm7_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res7_3)
    add7_3 = add([norm7_3, conv7_1])
    up7 = UpSampling2D(size=(2, 2))(add7_3)
    concat7 = concatenate([inputs[2], up7], axis=3)  # 64*64*768

    conv8_1 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(concat7)
    res8_1 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                    name='res8_1_' + input_type, kernel_initializer='he_normal')(conv8_1)
    norm8_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res8_1)
    res8_2 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same',
                    name='res8_2_' + input_type, kernel_initializer='he_normal')(norm8_1)
    norm8_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res8_2)
    res8_3 = Conv2D(filters=256, kernel_size=3, activation=None, padding='same',
                    name='res8_3_' + input_type, kernel_initializer='he_normal')(norm8_2)
    norm8_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res8_3)
    add8_3 = add([norm8_3, conv8_1])
    up8 = UpSampling2D(size=(2, 2))(add8_3)
    concat8 = concatenate([inputs[1], up8], axis=3)  # 128*128*384

    conv9_1 = Conv2D(filters=128, kernel_size=1, activation='relu', padding='same')(concat8)
    res9_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                    name='res9_1_' + input_type, kernel_initializer='he_normal')(conv9_1)
    norm9_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res9_1)
    res9_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                    name='res9_2_' + input_type, kernel_initializer='he_normal')(norm9_1)
    norm9_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res9_2)
    res9_3 = Conv2D(filters=128, kernel_size=3, activation=None, padding='same',
                    name='res9_3_' + input_type, kernel_initializer='he_normal')(norm9_2)
    norm9_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res9_3)
    add9_3 = add([norm9_3, conv9_1])
    up9 = UpSampling2D(size=(2, 2))(add9_3)
    concat9 = concatenate([inputs[0], up9], axis=3)  # 256*256*192

    conv10_1 = Conv2D(filters=128, kernel_size=1, activation='relu', padding='same')(concat9)
    res10_1 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='res10_1_' + input_type, kernel_initializer='he_normal')(conv10_1)
    norm10_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res10_1)
    res10_2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
                     name='res10_2_' + input_type, kernel_initializer='he_normal')(norm10_1)
    norm10_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res10_2)
    res10_3 = Conv2D(filters=128, kernel_size=3, activation=None, padding='same',
                     name='res10_3_' + input_type, kernel_initializer='he_normal')(norm10_2)
    norm10_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res10_3)
    add10_3 = add([norm10_3, conv10_1])
    encode_output = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                           name='encode_output' + input_type, kernel_initializer='he_normal')(add10_3)  # 256*256*64

    embeddings = tf.nn.l2_normalize(encode_output, axis=-1)

    return embeddings


def decoding(corr, inputs):
    pool11 = MaxPooling2D(pool_size=(16, 16), strides=(16, 16), name='pool11')(corr)  # 16*16*1
    conv11 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv11', kernel_initializer='he_normal')(inputs[4])  # 16*16*64 VGG features

    concat11 = concatenate([pool11, conv11], axis=3)  # 16*16*65

    conv11_1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(concat11)  # 1*1 conv
    res11_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res11_1', kernel_initializer='he_normal')(conv11_1)
    norm11_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res11_1)
    res11_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res11_2', kernel_initializer='he_normal')(norm11_1)
    norm11_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res11_2)
    res11_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res11_3', kernel_initializer='he_normal')(norm11_2)  # 16*16*64
    norm11_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res11_3)
    res11_3 = add([norm11_3, conv11_1])

    up12 = UpSampling2D(size=(2, 2))(res11_3)  # 32*32*64
    conv12 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv12', kernel_initializer='he_normal')(inputs[3])  # 32*32*64 VGG features
    pool12 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), name='pool12')(corr)  # 32*32*1
    concat12 = concatenate([up12, conv12, pool12], axis=3)  # 32*32*129

    conv12_1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(concat12)  # 1*1 conv
    res12_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res12_1', kernel_initializer='he_normal')(conv12_1)
    norm12_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res12_1)
    res12_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res12_2', kernel_initializer='he_normal')(norm12_1)
    norm12_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res12_2)
    res12_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res12_3', kernel_initializer='he_normal')(norm12_2)  # 32*32*64
    norm12_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res12_3)
    res12_3 = add([norm12_3, conv12_1])

    up13 = UpSampling2D(size=(2, 2))(res12_3)  # 64*64*64
    conv13 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv13', kernel_initializer='he_normal')(inputs[2])  # 64*64*64 VGG features
    pool13 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool13')(corr)  # 64*64*1
    concat13 = concatenate([up13, conv13, pool13], axis=3)  # 64*64*129

    conv13_1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(concat13)
    res13_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res13_1', kernel_initializer='he_normal')(conv13_1)
    norm13_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res13_1)
    res13_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res13_2', kernel_initializer='he_normal')(norm13_1)
    norm13_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res13_2)
    res13_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res13_3', kernel_initializer='he_normal')(norm13_2)  # 64*64*64
    norm13_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res13_3)
    res13_3 = add([norm13_3, conv13_1])

    up14 = UpSampling2D(size=(2, 2))(res13_3)  # 128*128*64
    conv14 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv14', kernel_initializer='he_normal')(inputs[1])  # 128*128*64 VGG features
    pool14 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool14')(corr)  # 128*128*1
    concat14 = concatenate([up14, conv14, pool14], axis=3)  # 128*128*129

    conv14_1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(concat14)
    res14_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res14_1', kernel_initializer='he_normal')(conv14_1)
    norm14_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res14_1)
    res14_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res14_2', kernel_initializer='he_normal')(norm14_1)
    norm14_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res14_2)
    res14_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res14_3', kernel_initializer='he_normal')(norm14_2)  # 128*128*64
    norm14_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res14_3)
    res14_3 = add([norm14_3, conv14_1])

    up15 = UpSampling2D(size=(2, 2))(res14_3)  # 256*256*64
    conv15 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same',
                    name='conv15', kernel_initializer='he_normal')(inputs[0])  # 256*256*64 VGG features 已经是64*64，应该可以省去
    # pool15 = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), name='pool15')(corr_input) # 128*128*1
    concat15 = concatenate([up15, conv15, corr], axis=3)  # 256*256*129

    conv15_1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(concat15)
    res15_1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res15_1', kernel_initializer='he_normal')(conv15_1)
    norm15_1 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res15_1)
    res15_2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
                     name='res15_2', kernel_initializer='he_normal')(norm15_1)
    norm15_2 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res15_2)
    res15_3 = Conv2D(filters=64, kernel_size=3, activation=None, padding='same',
                     name='res15_3', kernel_initializer='he_normal')(norm15_2)  # 256*256*64
    norm15_3 = BatchNormalization(epsilon=1e-3, momentum=0.999)(res15_3)
    res15_3 = add([norm15_3, conv15_1])
    prediction = Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding='same',
                        name='prediction', kernel_initializer='he_normal')(res15_3)  # 256*256*1
    return prediction


strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))

keras.backend.clear_session()
with strategy.scope():
    model = new_model()
    # for i, w in enumerate(model.weights): print(i, w.name)
    for i in range(len(model.layers)):
        model.layers[i]._handle_name = model.layers[i]._name + "_" + str(i)
    model.summary()
    # plot_model(model, to_file='pretrained_model.png', show_shapes=True, show_layer_names=True)

    optimizer = Adam(lr=0.005)  # lr=0.00005
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  # 'binary_crossentropy'

callbacks = [
    keras.callbacks.ModelCheckpoint("pretrained_result.h5", save_best_only=True)
]
# gt = np.load('D:\projects\OTS_reconstruct\\test_data\\train_gt_10.npy')
# gt = gt[np.newaxis, :]
gt = np.load("../../test_data/train_gt_2000.npy").astype(int)
gt = np.expand_dims(gt, axis=-1)
# query = np.load('D:\projects\OTS_reconstruct\\test_data\\train_query_10.npy')\
query = np.load('../../test_data/train_query_2000.npy').astype(int)
# query = query[np.newaxis, :]
# ref = np.load('D:\projects\OTS_reconstruct\\test_data\\train_ref_10.npy')
ref = np.load('../../test_data/train_ref_2000.npy').astype(int)
# ref = ref[np.newaxis, :]
history = model.fit(x=[query, ref],
                    y=gt,
                    validation_split=0.2,
                    batch_size=16,
                    epochs=7,
                    # callbacks=callbacks,
                    shuffle=True)

validation_query = np.load('../../test_data/validation_query_10.npy')
validation_ref = np.load('../../test_data/validation_ref_10.npy')
# validation_query = validation_query[np.newaxis, :]
# validation_ref = validation_ref[np.newaxis, :]
validation_query = (validation_query * 255).astype(int)
validation_ref = (validation_ref * 255).astype(int)

predict = model.predict(x=[validation_query, validation_ref])
for i in range(10):
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(validation_query[i, :, :, :])
    plt.subplot(132)
    plt.imshow(validation_ref[i, :, :, :])
    plt.subplot(133)
    plt.imshow(predict[i, :, :, 0])
    plt.show()
# Predict with the training data to test if the model behave properly
train_data_query = query[0:5, :, :, :]
train_data_ref = ref[0:5, :, :, :]
train_data_predict = model.predict(x=[train_data_query, train_data_ref])
for i in range(5):
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(train_data_query[i, :, :, :])
    plt.subplot(132)
    plt.imshow(train_data_ref[i, :, :, :])
    plt.subplot(133)
    plt.imshow(train_data_predict[i, :, :, 0])
    plt.show()
