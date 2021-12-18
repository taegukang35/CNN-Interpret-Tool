import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

model_builder = keras.applications.vgg16.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input
img_size = (224,224)
img_path = "dog.jpg"
img = cv2.imread(img_path)
img_array = preprocess_input(get_img_array(img_path, size=img_size))
model = model_builder(weights='imagenet')

layer_outputs = [layer.output for layer in model.layers]
# Returns the output of eight layers for the input.
activation_model = models.Model(inputs=model.input,outputs=layer_outputs)
activations = activation_model.predict(img_array)

plt.clf()
layer_names = []
for layer in model.layers[:len(model.layers)]:
  layer_names.append(layer.name)
images_per_row = 16

for layer_name, layer_activation in zip(layer_names,activations):
  n_features = layer_activation.shape[-1] # 특성 맵에 있는 특성의 수
  size = layer_activation.shape[1]  # (1,size,size,n_features)
  n_cols = n_features // images_per_row
  display_grid = np.zeros((size*n_cols,images_per_row*size))

  for col in range(n_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0,:,:,col*images_per_row+row]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image,0,255).astype('uint8')
      display_grid[col*size:(col+1)*size,row*size:(row+1)*size] = channel_image
  scale = 1./size
  plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid,aspect='auto',cmap='viridis')
  plt.show()
