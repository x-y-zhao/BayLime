import os
import time
import shutil
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

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

############################################################
# use heatmap from Grad-CAM as prior knowledge for BayLime #
############################################################

# here we create a standard InceptionV3 pretrained model
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

mkdir('evaluation_output')
fname = 'evaluation_output'

images = transform_img_fn([os.path.join('data','penguin.jpeg')])

# I'm dividing by 2 and adding 0.5 because of
# how this Inception represents images

deletion = evaluation.CausalMetric(inet_model,'del')
insertion = evaluation.CausalMetric(inet_model,'ins')


preds = inet_model.predict(images)
pred_label = decode_predictions(preds)[0]


time1 = time.time()

explainer = lime_image.LimeImageExplainer(feature_selection='none')#kernel_width=0.1

explanation = explainer.explain_instance(images[0], inet_model.predict,
                                         top_labels=1, hide_color=0, batch_size=15,
                                         num_samples=200, model_regressor='BayesianRidge_inf_prior_fit_alpha')

#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'

time2 = time.time()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=10, hide_rest=False)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
ax.set_ylabel(pred_label[0][1],fontsize=20)
fig.savefig(fname+'/Lime_exp.png',bbox_inches='tight')


h1_del = deletion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'Lime',fname)
h1_ins = insertion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'Lime',fname)


time3 = time.time()

# extract the prior knowledge from grad-cam
prior_knowledge = Grad_CAM.extrat_prior(images[0],inet_model,explanation,fname,pred_label[0][1])
prior_exp = np.flip(np.argsort(abs(np.array(prior_knowledge))))
seg = explanation.segments

time4 = time.time()

h2_del = deletion.single_run(images[0], prior_exp, seg, explanation.top_labels[0], pred_label, 'Grad_CAM',fname)
h2_ins = insertion.single_run(images[0], prior_exp, seg, explanation.top_labels[0], pred_label, 'Grad_CAM',fname)


time5 = time.time()

# update the explanation with prior
alpha_init=1
lambda_init=1
with open('./posterior_configure.csv') as csv_file:
    csv_reader=csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 1:
            alpha_init=float(row[0])
            lambda_init=float(row[1])
        line_count=line_count+1


explanation = calculate_posteriors.get_posterior(explanation,prior_knowledge,
                                               hyper_para_alpha=alpha_init,
                                               hyper_para_lambda=lambda_init,
                                               label=explanation.top_labels[0])

time6 = time.time()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=10, hide_rest=False)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
ax.set_ylabel(pred_label[0][1],fontsize=20)
fig.savefig(fname+'/BayLime_exp.png',bbox_inches='tight')

h3_del = deletion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'BayLime',fname)
h3_ins = insertion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'BayLime',fname)




print('-----------------------------')
print('Lime deletion: ', h1_del)
print('Lime insertion: ', h1_ins)
print('Grad-CAM deletion: ', h2_del)
print('Grad-CAM insertion: ', h2_ins)
print('Baylime deletion: ', h3_del)
print('Baylime insertion: ', h3_ins)
print('-----------------------------')
print("grad_CAM:", time4 - time3, "s")
print("Lime :", time2 - time1, "s")
print("BayLime :", time6 - time5 + time2 - time1 + time4 - time3, "s")

