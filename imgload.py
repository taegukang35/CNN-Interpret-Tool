import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras


img_path = keras.utils.get_file(
    "image.jpeg", "https://i.imgur.com/lJJZVRv.jpeg")
#img_path = "dog.jpg"

img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()