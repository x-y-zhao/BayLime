# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:45:00 2020

@author: XZ
"""

import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from skimage.segmentation import mark_boundaries
#from lime.lime_image import *
from lime import lime_image
print('Notebook run using keras:', keras.__version__)

#Here we create a standard InceptionV3 pretrained model 
#and use it on images by first preprocessing them with the preprocessing tools
inet_model = inc_net.InceptionV3()


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


images = transform_img_fn([os.path.join('data','frog.jpg')])
# I'm dividing by 2 and adding 0.5 because of
# how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
plt.show()
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)
    
    
    


explainer = lime_image.LimeImageExplainer(feature_selection='auto')#kernel_width=0.1

# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
explanation = explainer.explain_instance(images[0], inet_model.predict,
                                         top_labels=5, hide_color=0, 
                                         num_samples=1000,model_regressor='non_Bay')#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior'

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()