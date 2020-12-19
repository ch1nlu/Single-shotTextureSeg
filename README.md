# One-shotTextureSeg

## Project objective

The objective of our work is to build an one-shot texture segmentation deep learning pipeline which is robust to scale variance and rotation. The codes are implemented mainly in [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) API. We also provide the numpy code to generate the texture collage dataset, which is one of the training datasets we are using to evaluate the performance of our neural network. Our work is mainly based on the [OSTS](https://arxiv.org/abs/1807.02654) paper, and we are trying to improve the performance of its model by introducing scale-invariance and rotation-invariance features. 

In the 'model' folder, you will find the OSTS scripts that I wrote from scratch.

In the 'results' folder, you can find the visualization of our recent results and get a sense of what we are working on.

Primary references: [One-shot Texture Segmentation](https://arxiv.org/abs/1807.02654);[One-Shot Texture Retrieval with Global Context Metric](https://arxiv.org/abs/1905.06656)
