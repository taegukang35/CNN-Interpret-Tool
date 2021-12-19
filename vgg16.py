import tensorflow as tf
from tensorflow import keras
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap
#import tensorflow.compat.v1.keras.backend as K
#tf.compat.v1.disable_eager_execution()


model_builder = keras.applications.vgg16.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input
img_size = (224, 224)
model = model_builder(weights='imagenet')
model.layers[-1].activation = None

def get_img_array(img_path, size):
    with tf.device('/cpu:0'):
        img = keras.preprocessing.image.load_img(img_path, target_size=size)
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

def predict_class(img_path):
    with tf.device('/cpu:0'):
        img_array = preprocess_input(get_img_array(img_path, size=img_size))
        preds = model.predict(img_array)
        return keras.applications.vgg16.decode_predictions(preds,top=1)[0]

def get_featuremap(img_path,layer_name):
    with tf.device('/cpu:0'):
        img_array = preprocess_input(get_img_array(img_path, size=img_size))
        model = model_builder(weights='imagenet')
        layer_outputs = [layer.output for layer in model.layers]
        # Returns the output of eight layers for the input.
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(img_array)

        layer_names = []
        for layer in model.layers[:len(model.layers)]:
            layer_names.append(layer.name)
        images_per_row = 16
        for layer, layer_activation in zip(layer_names, activations):
            if layer == layer_name:
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1]  # (1,size,size,n_features)
                #n_cols = n_features // images_per_row
                n_cols = 4 # 출력 편의상 4로 고정
                display_grid = np.zeros((size * n_cols, images_per_row * size))

                for col in range(n_cols):
                    for row in range(images_per_row):
                        channel_image = layer_activation[0, :, :, col * images_per_row + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
                scale = 1. / size
                plt.imsave('feature_map.png',display_grid)


def get_gradcam(img_path,last_conv_layer_name):
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    heatmap = make_gradcam_heatmap(img_array,last_conv_layer_name)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # heatmap을 RGB 포맷으로 변환
    cv2.imwrite('heatmap.jpg', heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 히트맵으로 변환
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('gradcam.jpg', superimposed_img)


def make_gradcam_heatmap(img_array, last_conv_layer_name, pred_index=None):
    with tf.device('/cpu:0'):
        model = model_builder(weights='imagenet')
        model.layers[-1].activation = None
        grad_model = models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

def get_lime(img_path):
    def transform_img_fn(path_list):
        out = []
        for img_path in path_list:
            img = image.load_img(img_path, target_size=img_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            out.append(x)
        return np.vstack(out)
    images = transform_img_fn([img_path])
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images[0].astype('double'), model.predict, top_labels=5, hide_color=0,
                                             num_samples=500)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=8,
                                                hide_rest=True)
    lime = mark_boundaries(temp, mask)
    plt.imshow(mask)
    plt.savefig('lime_mask.jpg')
    cv2.imwrite('lime.jpg', lime)

"""
def get_shap(img_path):
    def map2layer(x, layer):
        feed_dict = dict(zip([model.layers[0].input], [preprocess_input(x.copy())]))
        return K.get_session().run(model.layers[layer].input, feed_dict)
    X, y = shap.datasets.imagenet50()
    to_explain = get_img_array(img_path, img_size)
    e = shap.GradientExplainer((model.layers[7].input, model.layers[-1].output),
                               map2layer(preprocess_input(X.copy()), 7))
    shap_values, indexes = e.shap_values(map2layer(to_explain, 7), ranked_outputs=2)
    shap.image_plot(shap_values, to_explain, show=False)
    plt.savefig('shap.jpg')
"""
