# BayLIME

BayLIME is a Bayesian modification of LIME that provides a principled mechanism to combine useful knowledge (e.g., from other diverse XAI methods, embedded human knowledge in the training of the AI/ML model under explanation or simply previous explanations of similar instances), which is a clear trend in AI. Such combination benefits the consistency in repeated explanations of a single prediction, robustness to kernel settings and may also improve the efficiency by requiring less queries made to the AI/ML model.

## Setup
1. Copy-paste the modified_sklearn_BayesianRidge.py file (in the lime/utils folder on this repo) into your local sklearn.linear_model folder.
2. To find out where the folder is, simply run:
```python
from sklearn import linear_model
print(linear_model.__file__)
```
(Tested with Python version 3.7.3, scikit-learn version 0.22.1, Tensorflow version 2.0.0)
3. Download the necessary dataset for ImageNet and GTSRB model, unzip the files and move to the data folder.
```
Insert download link here
```

## Repository Structure

* **experiments** contains the experiments for the draft paper, in which you may find both the code (in Python jupyter-notebook) and the original data generated (stored as HTML and .csv files).
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

Please refer to the tutorials (e.g., BayLIME_tutorial_images.ipynb)  for details.

## Embed Prior from GradCAM
To get the explanation for specified image (e.g. king penguin) in data folder, firstly modify the image path in Line 46 of Grad_CAM_Prior.py, then type
```
python Grad_CAM_Prior.py
```
You will get the explanation results along with Deletion and Insertion AUC figures from GradCAM, LIME and BayLIME under the created evaluation_output folder.
To get statistical fidelity evaluation on ImageNet dataset, firstly make sure the validation dataset from ImageNet called ILSVRC2012_img_val is already downloaded and moved to the data folder, then type
```
python del_ins_exp.py
```
You will the explanation result for each image from ImageNet and a record file for recording the runtime output in the created evaluation_output folder. Be cautious the evaluation_output folder will be reset every time running the program, so take a copy if you want to save the running results.

## Embed Prior from Neural Cleanse
In backdoor_exp.py, we provide the explanation for backdoor input from BadNet and TrojanAttack models. To get the IoU and AMD evaluation for Prior, LIME and BayLIME, type the command
```
python backdoor_exp.py
```




