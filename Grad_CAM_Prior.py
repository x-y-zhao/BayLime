import os
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



images = transform_img_fn([os.path.join('data','1.jpg')])

# I'm dividing by 2 and adding 0.5 because of
# how this Inception represents images

deletion = evaluation.CausalMetric(inet_model,'del')
insertion = evaluation.CausalMetric(inet_model,'ins')

plt.imshow(images[0] / 2 + 0.5)
plt.show()
preds = inet_model.predict(images)
pred_label = decode_predictions(preds)[0]
# for x in decode_predictions(preds)[0]:
#     print(x)

explainer = lime_image.LimeImageExplainer(feature_selection='none')#kernel_width=0.1

explanation = explainer.explain_instance(images[0], inet_model.predict,
                                         top_labels=1, hide_color=0, batch_size=15,
                                         num_samples=2000,model_regressor='Bay_info_prior')
#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior','BayesianRidge_inf_prior_fit_alpha'


h = deletion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'Lime')
h = insertion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'Lime')

# extract the prior knowledge from grad-cam
prior_knowledge = Grad_CAM.extrat_prior(images,inet_model,explanation)


prior_exp = np.flip(np.argsort(abs(np.array(prior_knowledge))))
seg = explanation.segments


h = deletion.single_run(images[0], prior_exp, seg, explanation.top_labels[0], pred_label, 'Grad_CAM')
h = insertion.single_run(images[0], prior_exp, seg, explanation.top_labels[0], pred_label, 'Grad_CAM')


# update the explanation with prior
alpha_var=1
lambda_var=5

explanation=calculate_posteriors.get_posterior(explanation,prior_knowledge,
                                               hyper_para_alpha=alpha_var,
                                               hyper_para_lambda=lambda_var,
                                               label=explanation.top_labels[0])


h = deletion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'BayLime')
h = insertion.single_run(images[0], explanation.local_exp[explanation.top_labels[0]], explanation.segments, explanation.top_labels[0], pred_label, 'BayLime')


temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)


plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

print(explanation.as_list(explanation.top_labels[0]))