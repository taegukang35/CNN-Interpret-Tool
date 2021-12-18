import random
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries


model_builder = keras.applications.vgg16.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input
img_size = (224, 224)
model = model_builder(weights='imagenet')
model.layers[-1].activation = None

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path,target_size=img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        out.append(x)
    return np.vstack(out)

#img_path = keras.utils.get_file(str(random.random())+"image.jpeg","https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQigoOMF47ZhIimseAc9hmw3oK6YRuNuafGqw&usqp=CAU")
img_path = 'dog.jpg'

images = transform_img_fn([img_path])
img = cv2.imread(img_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
preds = model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=500)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=8, hide_rest=True)
lime = mark_boundaries(temp, mask)

from skimage.color import label2rgb
# 캔버스 설정
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(8,8))

# 예측에 도움이 되는 세그먼트 출력
temp,mask = explanation.get_image_and_mask(explanation.top_labels[0],positive_only=True,num_features=8,hide_rest=False)
ax1.imshow(label2rgb(mask,temp,bg_label=0),interpolation='nearest')
ax1.set_title('Positive Region for {}'.format(explanation.top_labels[0]))

# 모든 세그먼트 출력
temp,mask = explanation.get_image_and_mask(explanation.top_labels[0],positive_only=False,num_features=8,hide_rest=False)
ax2.imshow(label2rgb(4-mask,temp,bg_label=0),interpolation='nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(explanation.top_labels[0]))

# 이미지만 출력
ax3.imshow(temp,interpolation='nearest')
ax3.set_title('Show output image only')

# 마스크만 출력
ax4.imshow(mask,interpolation='nearest')
ax4.set_title('Show mask only')
plt.show()