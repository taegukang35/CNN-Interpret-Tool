import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

model_builder = keras.applications.vgg16.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input
img_size = (224,224)

#img_path = keras.utils.get_file(
#    "input_img", "https://i.imgur.com/1o3KcN6.png")
img_path = "dog.jpg"

img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()
img_array = preprocess_input(get_img_array(img_path, size=img_size))

model = model_builder(weights='imagenet')
print(model.summary())
model.layers[-1].activation = None
preds = model.predict(img_array)
print("Predictred:",keras.applications.vgg16.decode_predictions(preds,top=1)[0])
last_conv_layer_name = 'block5_conv3'
print(last_conv_layer_name)
heatmap = make_gradcam_heatmap(img_array,model,last_conv_layer_name)
plt.imshow(heatmap,cmap='jet',alpha=.5)
plt.show()

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255*heatmap) #heatmap을 RGB 포맷으로 변환
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET) #히트맵으로 변환
superimposed_img = heatmap*0.4 + img

cv2.imwrite('grad_cam.jpg',superimposed_img)
im = Image.open('grad_cam.jpg')
im.show()
