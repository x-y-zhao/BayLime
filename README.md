# BayLIME
This is an anonymous website for our AAAI2021 submission.

BayLIME is a Bayesian modification of LIME that provides a principled mechanism to combine useful knowledge (e.g., from other diverse XAI methods, embedded human knowledge in the training of the AI/ML model under explanation or simply previous explanations of similar instances), which is a clear trend in AI. Such combination benefits the consistency in repeated explanations of a single prediction, robustness to kernel settings and may also improve the efficiency by requiring less queries made to the AI/ML model.

## Setup
1. Copy-paste the modified_sklearn_BayesianRidge.py file (in the lime/utils folder on this repo) into your local sklearn.linear_model folder.
2. To find out where the folder is, simply run:
```python
from sklearn import linear_model
print(linear_model.__file__)
```
(Tested with Python version 3.7.3 and scikit-learn version 0.22.1)

## Repository Structure

* **AAAI21_experiments** contains the experiments for the AAAI submission, in which you may find both the code (in Python jupyter-notebook) and the original data generated (stored as HTML and .csv files).
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

Please refer to the tutorials for details.
