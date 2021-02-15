import os
import time
import shutil
from os import listdir
from os.path import isfile, join
import shap
from lime.wrappers.scikit_image import SegmentationAlgorithm
import tensorflow.keras
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.utils.record import record, shap_writeInfo
from lime import lime_image
from lime import Grad_CAM
from lime import evaluation
import csv
from lime import calculate_posteriors
print('Notebook run using keras:', tensorflow.keras.__version__)


############################################################
# use heatmap from Grad-CAM as prior knowledge for BayLime #
############################################################
# some necessary functions
# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))

    # Create an empty 4D array
    out = np.zeros((zs.shape[0],
                    image.shape[0],
                    image.shape[1],
                    image.shape[2]))

    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    return out




def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

############################################################

# import model trained with imagenet
inet_model = inc_net.InceptionV3(weights='imagenet')

# set record file
mkdir('evaluation_output')
r = record('evaluation_output/shap_record.txt',time.time())

# import imagenet data from file
dataset_path = 'data/ILSVRC2012_img_val'
images_paths = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
images = transform_img_fn(images_paths[:3])

# visualize some images
# plt.imshow(images[1] / 2 + 0.5)
# plt.show()


# initialize the evaluation with insert and delete algorithm
deletion = evaluation.CausalMetric(inet_model,'del')
insertion = evaluation.CausalMetric(inet_model,'ins')


preds = inet_model.predict(images)
pred_label = decode_predictions(preds)
top_preds = np.argsort(-preds)


ins_shap = []
del_shap = []


for i in range(1000):

    print("---------------------")
    print('Image No. ', i)
    print("---------------------")
    segmentation_fn = SegmentationAlgorithm('slic', kernel_size=4,
                                            max_dist=200, ratio=0.2,
                                            random_seed=None)
    segments = segmentation_fn(images[i])

    n_segments = np.max(segments)+1

    def f(z):
        return inet_model.predict(mask_image(z, segments, images[i], None))

    #'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'

    aa = time.time()
    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1, n_segments)))


    shap_values = explainer.shap_values(np.ones((1, n_segments)), nsamples=200)

    results = shap_values[top_preds[i][0]]

    exp_results = np.flip(np.argsort(abs(np.array(results))))

    bb = time.time()
    aaa = bb - aa



    # create folder to save output file
    fname = "evaluation_output/image_" + str(i)
    mkdir(fname)

    h1_del = deletion.single_run(images[i], exp_results[0], segments, top_preds[i][0], pred_label[i], 'SHAP',fname)
    h1_ins = insertion.single_run(images[i], exp_results[0], segments, top_preds[i][0], pred_label[i], 'SHAP',fname)


    ins_shap.append(h1_ins)
    del_shap.append(h1_del)

    shap_writeInfo(r, i, h1_del, h1_ins)

    print("---------------------")
    print('Run Time: ', aaa)
    print('Shap deletion: ', np.mean(del_shap))
    print('Shap insertion: ', np.mean(ins_shap))
    print("---------------------")

shap_writeInfo(r, -1, np.mean(del_shap), np.mean(ins_shap))
r.close()
