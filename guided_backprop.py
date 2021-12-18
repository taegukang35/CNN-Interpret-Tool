import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input,decode_predictions
from tensorflow.python.framework import ops
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
tf.compat.v1.disable_eager_execution()

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_image(path, target_size=(224, 224)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

with tf.Graph().as_default() as g:
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    with g.gradient_override_map({"Relu": "GuidedRelu"}):
        # Do stuff here


def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights='imagenet')
    return new_model


def guided_backpropagation(img_tensor, model, activation_layer):
    model_input = model.input
    layer_output = model.get_layer(activation_layer).output

    # one_output = layer_output[:, :, :, 256]
    max_output = K.max(layer_output, axis=3)

    get_output = K.function([model_input], [K.gradients(max_output, model_input)[0]])
    # get_output = K.function([model_input], [K.gradients(one_output, model_input)[0]])
    saliency = get_output([img_tensor])

    return saliency[0]

with tf.device('/CPU:0'):
    img_width = 224
    img_height = 224
    img_path = 'dog.jpg'
    img = load_image(path=img_path, target_size=(img_width, img_height))

    model = VGG16(weights='imagenet')

    preds = model.predict(img)
    predicted_class = preds.argmax(axis=1)[0]
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print("predicted top1 class:", predicted_class)
    print('Predicted:',decode_predictions(preds, top=1)[0])
    conv_name = 'block5_conv3'

    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    gradient = guided_backpropagation(img, guided_model, conv_name)

    plt.imshow(deprocess_image(gradient))
    plt.show()