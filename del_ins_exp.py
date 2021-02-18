import os
import time
import shutil
from os import listdir
from os.path import isfile, join
import tensorflow.keras
from tensorflow.keras.applications import inception_v3 as inc_net
# from tensorflow.keras.applications import resnet50 as inc_net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.utils.record import record, writeInfo
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
# inet_model = inc_net.ResNet50(weights='imagenet')
# inet_model.summary()

# set record file
mkdir('evaluation_output')
r = record('evaluation_output/record.txt',time.time())

# import imagenet data from file
dataset_path = 'data/ILSVRC2012_img_val'
images_paths = [join(dataset_path, f) for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
images = transform_img_fn(images_paths[:30])

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

for i in range(500, 1000):


    print("---------------------")
    print('Image No. ', i)
    print("---------------------")
    explanation = explainer.explain_instance(images[i], inet_model.predict,
                                            top_labels=1, hide_color=0, batch_size=15,
                                            num_samples=200, model_regressor='BayesianRidge_inf_prior_fit_alpha')
    #'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'

    # create folder to save output file
    fname = "evaluation_output/image_" + str(i)
    mkdir(fname)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax.set_ylabel(pred_label[i][0][1],fontsize=20)
    fig.savefig(fname+'/Lime_exp.png',bbox_inches='tight')

    h1_del = deletion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'Lime',fname)
    h1_ins = insertion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'Lime',fname)

    ins_lime.append(h1_ins)
    del_lime.append(h1_del)


    # extract the prior knowledge from grad-cam
    prior_knowledge = Grad_CAM.extrat_prior(images[i],inet_model,explanation,fname,pred_label[i][0][1])
    prior_exp = np.flip(np.argsort(abs(np.array(prior_knowledge))))
    seg = explanation.segments


    h2_del = deletion.single_run(images[i], prior_exp, seg, explanation.top_labels[0], pred_label[i], 'Grad_CAM',fname)
    h2_ins = insertion.single_run(images[i], prior_exp, seg, explanation.top_labels[0], pred_label[i], 'Grad_CAM',fname)

    ins_gcam.append(h2_ins)
    del_gcam.append(h2_del)


    # update the explanation with prior
    # dynamicly adjust lambda_var
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

    explanation=calculate_posteriors.get_posterior(explanation,prior_knowledge,
                                                hyper_para_alpha=alpha_init,
                                                hyper_para_lambda=lambda_init,
                                                label=explanation.top_labels[0])


    h3_del = deletion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'BayLime',fname)
    h3_ins = insertion.single_run(images[i], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label[i], 'BayLime',fname)

    ins_blime.append(h3_ins)
    del_blime.append(h3_del)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=3, hide_rest=False)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax.set_ylabel(pred_label[i][0][1],fontsize=20)
    fig.savefig(fname+'/BayLime_exp.png',bbox_inches='tight')



    writeInfo(r, i, h1_del, h1_ins, h2_del, h2_ins, h3_del, h3_ins)

    print("---------------------")
    print('Lime deletion: ', np.mean(del_lime))
    print('Lime insertion: ', np.mean(ins_lime))
    print('Grad-CAM deletion: ', np.mean(del_gcam))
    print('Grad-CAM insertion: ', np.mean(ins_gcam))
    print('Baylime deletion: ', np.mean(del_blime))
    print('Baylime insertion: ', np.mean(ins_blime))
    print("---------------------")

writeInfo(r, -1, np.mean(del_lime), np.mean(ins_lime), np.mean(del_gcam), np.mean(ins_gcam), np.mean(del_blime), np.mean(ins_blime))
r.close()

# temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
# plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
# plt.show()
# print(explanation.as_list(explanation.top_labels[0]))
