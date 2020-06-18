# BayLime (Bayesian Lime)

The gist of BayLime is to allow users to embed *informative* prior knowledge when interpreting AI using Lime.

I have modified the Lime by adding more options to the args *model_regressor*.

Now when calling the explainer.explain_instance() API of BayLime:
1. model_regressor='non_Bay' (default) uses sklearn Ridge regressor
2. model_regressor='Bay_non_info_prior' uses sklearn BayesianRidge regressor with all default args (fitting both hyperparameters alpha and lambda from samples)
3. model_regressor='Bay_info_prior' uses XZ modified sklearn BayesianRidge regressor and reads the hyperparameters alpha and lambda from configuration files, 
4. model_regressor='BayesianRidge_inf_prior_fit_alpha' uses XZ modifed BayesianRidge regressor and reads the hyperparameters lambda from configuration files and fit alpha from sampling data.

Te point 3 and 4 requires to copy-paste the modified_sklearn_BayesianRidge.py file (in the lime/utils folder on this repo) into your local sklearn.linear_model folder.

To find out where the folder is, simply run:

```python
from sklearn import linear_model
print(linear_model.__file__)
```


Then, simply run the test_tabular.py. Or run Jupyter Notebook file experiment_tabular_boston_dataset.ipynb. 

(Tested with Python version 3.7.3 and scikit-learn version 0.22.1)

