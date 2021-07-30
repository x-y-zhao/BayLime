# BayLIME

BayLIME is a Bayesian modification of [LIME](https://github.com/marcotcr/lime) (one of the most widely used approaches in XAI). Compared to LIME, BayLIME exploits prior knowledge and Bayesian reasoning to improve both the consistency in repeated explanations of a single prediction and the robustness to kernel settings. BayLIME also exhibits better explanation fidelity than the state-of-the-art (LIME, SHAP and GradCAM) by its ability to integrate prior knowledge from, e.g., a variety of other XAI techniques, as well as verification and validation (V&V) methods.

## Publication

The paper on BayLIME is accepted by UAI2021, here is the [accepted version](https://arxiv.org/pdf/2012.03058.pdf) on arXiv and a [recorded presentation](https://youtu.be/-ftJFNVxvOQ) for UAI'21.

## Setup
1. Copy-paste the modified_sklearn_BayesianRidge.py file (in the lime/utils folder on this repo) into your local sklearn.linear_model folder. To find out where the folder is, simply run:
```python
from sklearn import linear_model
print(linear_model.__file__)
```
2. Download the necessary dataset for ImageNet and GTSRB model, unzip the files and move to the data folder.
```
ImageNet (original images): http://image-net.org/download-images
GTSRB (.h5 file): https://drive.google.com/file/d/1MjgsnH3bOYG_PvdvqmoamoCPmySQazRJ/view?usp=sharing
```
(Tested with Python version 3.7.3, **scikit-learn version 0.22.1**, Tensorflow version 2.0.0)
## Repository Structure

* **experiments** contains the experiments of the paper, in which you may find both the code (in Python jupyter-notebook) and the original data generated (stored as HTML and .csv files).
* **lime**, all source-code of BayLIME that modifies the original LIME source-code can be found in this folder.
* **data** contains some data, e.g., images and tabular dataset.

## Basic Use

The gist of BayLime is to allow users to embed *informative* prior knowledge when interpreting AI using Lime.

We have modified the Lime by adding more options to the args *model_regressor*.

Now when calling the explainer.explain_instance() API of BayLime, we have four options:
1. model_regressor='non_Bay' (default) uses sklearn Ridge regressor
2. model_regressor='Bay_non_info_prior' uses sklearn BayesianRidge regressor with all default args (fitting both hyperparameters alpha and lambda from samples)
3. model_regressor='Bay_info_prior' uses the modified sklearn BayesianRidge regressor and reads the hyperparameters alpha and lambda from configuration files, 
4. model_regressor='BayesianRidge_inf_prior_fit_alpha' uses the modified BayesianRidge regressor and reads the hyperparameters lambda from configuration files and fit alpha from sampling data.

Please refer to the tutorials (e.g., **BayLIME_tutorial_images.ipynb**)  for details.

## Embed Prior from GradCAM
To get the explanation for a specific image (e.g. king penguin) in the data folder, firstly modify the image path in Line 46 of Grad_CAM_Prior.py, then type
```
python Grad_CAM_Prior.py
```
You will get the explanation results along with Deletion and Insertion AUC figures from GradCAM, LIME and BayLIME under the created *evaluation_output* folder.

To get statistical fidelity evaluation on ImageNet dataset, first please make sure the validation dataset from ImageNet called ILSVRC2012_img_val is already downloaded and moved to the data folder, then type
```
python del_ins_exp.py
```
You will get the explanation result for each image from ImageNet and a record file recording the runtime output in the created *evaluation_output* folder. Be cautious that the evaluation_output folder will be reset every time running the program, so take a copy if you want to save the results.

## Embed Prior from Neural Cleanse
In backdoor_exp.py, we provide the explanations for backdoor inputs based on BadNet and TrojanAttack models. To get the IoU and AMD evaluations for the Prior, LIME and BayLIME, type the command
```
python backdoor_exp.py
```
You will get the print out of IoU and AMD scores for each backdoor attacked images. The interpretation of IoU and AMD scores can be referred to the paper.




