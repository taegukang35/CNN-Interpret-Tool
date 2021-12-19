import argparse
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow.compat.v1.keras.backend as K
tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str)
parser.add_argument('--layer_name',type=str)
config = parser.parse_args()

model_builder = keras.applications.vgg16.VGG16
model = model_builder(weights = 'imagenet')
preprocess_input = keras.applications.vgg16.preprocess_input
img_size = (224, 224)

def layer_index(layer_name):
    if layer_name == 'block1_conv1': return 1
    elif layer_name == 'block1_conv2': return 2
    elif layer_name == 'block2_conv1': return 4
    elif layer_name == 'block2_conv2': return 5
    elif layer_name == 'block3_conv1': return 7
    elif layer_name == 'block3_conv2': return 8
    elif layer_name == 'block3_conv3': return 9
    elif layer_name == 'block4_conv1': return 11
    elif layer_name == 'block4_conv2': return 12
    elif layer_name == 'block4_conv3': return 13
    elif layer_name == 'block5_conv1': return 15
    elif layer_name == 'block5_conv2': return 16
    elif layer_name == 'block5_conv3': return 17


def get_img_array(img_path, size):
    with tf.device('/cpu:0'):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

index = layer_index(config.layer_name)

# load pre-trained model and choose two images to explain
model = VGG16(weights='imagenet', include_top=True)
X,y = shap.datasets.imagenet50()
to_explain = get_img_array(config.path,img_size)
# explain how the input to the layer of the model explains the top two classes
def map2layer(x, layer):
    feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
    return K.get_session().run(model.layers[layer].input, feed_dict)
e = shap.GradientExplainer((model.layers[index].input,
                            model.layers[-1].output),
                           map2layer(preprocess_input(X.copy()), index))
shap_values,indexes = e.shap_values(map2layer(to_explain, index), ranked_outputs=2)
shap.image_plot(shap_values,to_explain,show=False)
plt.savefig('shap.jpg')
