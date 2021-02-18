import os
import time
import shutil
import copy
import tensorflow.keras
import math
from lime.utils.generic_utils import cal_iou, cal_dist
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.spatial import distance
from PIL import Image, ImageOps
from backdoor.GTSRB import GTRSRB
from backdoor.trojannet import TrojanNet
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
import csv
from lime import calculate_posteriors
from tensorflow.keras.utils import to_categorical

print('Notebook run using keras:', tensorflow.keras.__version__)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


patch = np.zeros((6,6,3),dtype=int)
patch[0,0,:] = [82,107,17]
patch[0,1,:] = [176,0,20]
patch[0,2,:] = [240,156,60]
patch[0,3,:] = [249,157,200]
patch[0,4,:] = [208,233,109]
patch[0,5,:] = [34,21,151]
patch[1,0,:] = [110,147,202]
patch[1,1,:] = [250,175,58]
patch[1,2,:] = [114,90,139]
patch[1,3,:] = [146,154,39]
patch[1,4,:] = [41,24,59]
patch[1,5,:] = [68,0,227]
patch[2,0,:] = [193,60,220]
patch[2,1,:] = [204,193,164]
patch[2,2,:] = [153,115,126]
patch[2,3,:] = [183,137,79]
patch[2,4,:] = [221,20,121]
patch[2,5,:] = [111,181,113]
patch[3,0,:] = [177,104,251]
patch[3,1,:] = [109,222,94]
patch[3,2,:] = [22,84,44]
patch[3,3,:] = [107,176,221]
patch[3,4,:] = [37,179,198]
patch[3,5,:] = [127,156,131]
patch[4,0,:] = [137,88,251]
patch[4,1,:] = [42,132,152]
patch[4,2,:] = [229,201,159]
patch[4,3,:] = [26,84,97]
patch[4,4,:] = [170,209,135]
patch[4,5,:] = [78,182,27]
patch[5,0,:] = [125,255,160]
patch[5,1,:] = [132,178,88]
patch[5,2,:] = [22,14,15]
patch[5,3,:] = [141,128,64]
patch[5,4,:] = [29,148,80]
patch[5,5,:] = [61,52,102]



net_model = 'TrojanAttack'

if net_model == 'BadNet':
    gtrsrb = GTRSRB()
    gtrsrb.cnn_model()
    gtrsrb.load_model(name='gtsrb_bottom_right_white_4_target_33.h5')
    model = gtrsrb.model
    trigger_mask = np.zeros((32, 32), dtype=int)
    trigger_mask[27:31, 27:30] = 1
    n_s = 1200
    n_f = 16

elif net_model == 'TrojanAttack':
    gtrsrb = GTRSRB()
    gtrsrb.cnn_model()
    gtrsrb.load_model(name='GTSRB_trojan.h5')
    model = gtrsrb.model
    trigger_mask = np.zeros((32, 32), dtype=int)
    trigger_mask[24:30,24:30] = 1
    n_s = 500
    n_f = 9



elif net_model == 'TrojanNet':
    gtrsrb = GTRSRB()
    gtrsrb.cnn_model()
    gtrsrb.load_model(name='GTSRB.h5')

    backnet = TrojanNet()
    backnet.attack_left_up_point = (27, 27)
    backnet.synthesize_backdoor_map(all_point=16, select_point=5)
    backnet.trojannet_model()
    backnet.load_model('trojannet.h5')

    backnet.combine_model(target_model=gtrsrb.model, input_shape=(32, 32, 3), class_num=43, amplify_rate=2)
    model = backnet.backdoor_model
    image_pattern = backnet.get_inject_pattern(class_num=33)

    trigger_mask = np.zeros((32, 32), dtype=int)
    trigger_mask[backnet.attack_left_up_point[0]:backnet.attack_left_up_point[0] + 4,
    backnet.attack_left_up_point[1]:backnet.attack_left_up_point[1] + 4] = 1



print('loading dataset')
X_train, Y_train, X_test, Y_test = gtrsrb.load_dataset('data/gtsrb_dataset.h5')

print('creating backdoor dataset')
X_backdoor = copy.deepcopy(X_test).astype(float)

X_backdoor[:,24:30, 24:30,:] = patch

Y_backdoor = to_categorical([0] * len(X_backdoor), 43)


ground_truth = np.zeros((32, 32, 3), dtype=int)
ground_truth[24:30,24:30,:] = patch



# plt.imshow(X_backdoor[2600]/255)
# plt.show()
# #
# AA = model.predict(X_backdoor[2000:3000])

print('loading prior map')
im1 = Image.open('backdoor/gtsrb_trojanattack_0.png')
im2 = ImageOps.grayscale(im1)
prior_img = np.array(im2)
# prior_img = copy.deepcopy(trigger_mask)
# plt.imshow(img/255)
# plt.show()

# iou = cal_iou(prior_img, trigger_mask)
# amd = cal_dist(prior_img, trigger_mask)
#
# print(iou)
# print(amd)



grid_x, grid_y = np.mgrid[0:32, 0:32]

idx = np.nonzero(prior_img)
values = np.ones(len(idx[0]))

rbf = Rbf(idx[0], idx[1], values, epsilon=5, function='gaussian')  # inverse gaussian
prior = rbf(grid_x, grid_y)

# plt.imsave('p_a.png',prior)


# # evaluate the model
# loss1, acc1 = model.evaluate(X_test, Y_test, verbose=0)
# loss2, acc2 = model.evaluate(X_backdoor, Y_backdoor, verbose=0)


explainer = lime_image.LimeImageExplainer(feature_selection='none')  # kernel_width=0.1

l_iou_list = []
b_iou_list = []
l_amd_list = []
b_amd_list = []

mkdir('evaluation_output')

for i in range(500):

    print('-----------------------------------')
    print('-----------------------------------')
    print('No.:', i)

    explanation = explainer.explain_instance(X_backdoor[i], model.predict,
                                             top_labels=1, hide_color=0, batch_size=15, segmentation_fn='block',
                                             num_samples= n_s, model_regressor='BayesianRidge_inf_prior_fit_alpha')

    temp, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=n_f, hide_rest=False)

    iou = cal_iou(lime_mask, trigger_mask)
    amd = cal_dist(lime_mask, trigger_mask)


    # get the prior for each segments
    seg_prior = []
    seg_n = np.max(explanation.segments) + 1

    for j in range(seg_n):
        mask = np.where(explanation.segments == j, 0, explanation.segments)
        mask = np.where(explanation.segments != j, 1, mask)
        seg_prior.append(np.ma.array(prior, mask=mask).mean())

    prior_exp = np.flip(np.argsort(abs(np.array(seg_prior))))

    prior_mask = np.zeros(explanation.segments.shape, explanation.segments.dtype)
    for idx in range(16):
        prior_mask[explanation.segments == prior_exp[idx]] = 1

    prior_iou = cal_iou(prior_mask,trigger_mask)
    prior_amd = cal_dist(prior_mask,trigger_mask)

    # update the explanation with prior
    alpha_init = 1
    lambda_init = 1
    with open('./posterior_configure.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 1:
                alpha_init = float(row[0])
                lambda_init = float(row[1])
            line_count = line_count + 1

    explanation = calculate_posteriors.get_posterior(explanation, seg_prior,
                                                     hyper_para_alpha=alpha_init,
                                                     hyper_para_lambda=lambda_init,
                                                     label=explanation.top_labels[0])

    temp, baylime_mask = explanation.get_image_and_mask(explanation.top_labels[0], num_features=n_f, hide_rest=False)


    baylime_iou = cal_iou(baylime_mask,trigger_mask)
    baylime_amd = cal_dist(baylime_mask,trigger_mask)

    # create folder to save output file
    fname = "evaluation_output/image_" + str(i)
    mkdir(fname)

    plt.imsave(fname + '/prior.png',prior_mask)
    plt.imsave(fname + '/lime.png',lime_mask)
    plt.imsave(fname + '/baylime.png',baylime_mask)



    l_iou_list.append(iou)
    l_amd_list.append(amd)
    b_iou_list.append(baylime_iou)
    b_amd_list.append(baylime_amd)

    print('iou for prior:', prior_iou)
    print('amd for prior:', prior_amd)
    print('iou for lime:', iou)
    print('amd for lime:', amd)
    print('iou for baylime:', baylime_iou)
    print('amd for baylime:', baylime_amd)

print(np.mean(l_iou_list))
print(np.mean(l_amd_list))
print(np.mean(b_iou_list))
print(np.mean(b_amd_list))

