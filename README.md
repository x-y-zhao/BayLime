# BayLime (Bayesian Lime)

The gist of BayLime is to allow users to embed *informative* prior knowledge when interpreting AI using Lime.

I have modified the Lime by adding more options to the args *model_regressor*.

Now when calling the explainer.explain_instance() API of Lime:
1. model_regressor='non_Bay' (default) uses sklearn Ridge regressor
2. model_regressor='Bay_non_info_prior' uses sklearn BayesianRidge regressor with all default args
3. model_regressor='Bay_info_prior' uses XZ modified sklearn BayesianRidge regressor, 

The point 3 above requires to copy-paste the modified_sklearn_BayesianRidge.py file into your local sklearn.linear_model folder.

To find out where the folder is, simply run:

```python
from sklearn import linear_model
print(linear_model.__file__)
```

Then, simply run the test.py. 

