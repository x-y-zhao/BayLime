import os
from os import listdir
from os.path import isfile, join
import tensorflow.keras
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime import lime_image
from lime import Grad_CAM
from lime import evaluation
import csv
from lime import calculate_posteriors
print('Notebook run using keras:', tensorflow.keras.__version__)


############################################################
# use heatmap from Grad-CAM as prior knowledge for BayLime #
############################################################

# import model trained with imagenet
inet_model = inc_net.InceptionV3(weights='imagenet')

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


# import imagenet data from file
dataset_path = 'data/ILSVRC2012_img_val'
images_paths = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
images = transform_img_fn(images_paths[:100])

# visualize some images
# plt.imshow(images[1] / 2 + 0.5)
# plt.show()

# initialize the evaluation with insert and delete algorithm
deletion = evaluation.CausalMetric(inet_model,'del')
insertion = evaluation.CausalMetric(inet_model,'ins')


preds = inet_model.predict(images)
pred_label = decode_predictions(preds)


explainer = lime_image.LimeImageExplainer(feature_selection='none')#kernel_width=0.1

ins_lime = []
del_lime = []
ins_gcam = []
del_gcam = []
ins_blime = []
del_blime = []

for i in range(10):
    explanation = explainer.explain_instance(images[i], inet_model.predict,
                                            top_labels=1, hide_color=0, batch_size=15,
                                            num_samples=100, model_regressor='Bay_info_prior')
    #'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'


    h_del = deletion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'Lime')
    h_ins = insertion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'Lime')

    ins_lime.append(h_ins)
    del_lime.append(h_del)

    # extract the prior knowledge from grad-cam
    prior_knowledge = Grad_CAM.extrat_prior(images[i],inet_model,explanation)
    prior_exp = np.flip(np.argsort(abs(np.array(prior_knowledge))))
    seg = explanation.segments


    h_del = deletion.single_run(images[i], prior_exp, seg, explanation.top_labels[0], pred_label[i], 'Grad_CAM')
    h_ins = insertion.single_run(images[i], prior_exp, seg, explanation.top_labels[0], pred_label[i], 'Grad_CAM')

    ins_gcam.append(h_ins)
    del_gcam.append(h_del)


    # update the explanation with prior
    alpha_var=1
    lambda_var=5

    explanation=calculate_posteriors.get_posterior(explanation,prior_knowledge,
                                                hyper_para_alpha=alpha_var,
                                                hyper_para_lambda=lambda_var,
                                                label=explanation.top_labels[0])


    h_del = deletion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'BayLime')
    h_ins = insertion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'BayLime')

    ins_blime.append(h_ins)
    del_blime.append(h_del)

print('Lime deletion: ', np.mean(del_lime))
print('Lime insertion: ', np.mean(ins_lime))
print('Grad-CAM deletion: ', np.mean(del_gcam))
print('Grad-CAM insertion: ', np.mean(ins_gcam))
print('Baylime deletion: ', np.mean(del_blime))
print('Baylime insertion: ', np.mean(ins_blime))


# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
# plt.show()
# print(explanation.as_list(explanation.top_labels[0]))