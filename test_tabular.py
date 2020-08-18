# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:37:59 2020

@author: XZ
"""

from sklearn.datasets import load_boston,load_breast_cancer
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(1)
import lime
import lime.lime_tabular
from lime import submodular_pick
from lime import calculate_posteriors
import csv


#load example dataset
boston = load_boston()
cancer= load_breast_cancer()

#print a description of the variables
print(boston.DESCR)
print(cancer.DESCR)

#train a regressor
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target, train_size=0.80, test_size=0.20)
rf.fit(train, labels_train);

#train a classifier
# rf = sklearn.ensemble.RandomForestClassifier(n_estimators=1000)
# train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, train_size=0.80, test_size=0.20)
# rf.fit(train, labels_train);



# generate an "explainer" object
categorical_features  = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()


explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, 
                                                    verbose=False, mode='regression',discretize_continuous=False,feature_selection='none',sample_around_instance=False)

# categorical_features = np.argwhere(np.array([len(set(cancer.data[:,x])) for x in range(cancer.data.shape[1])]) <= 10).flatten()

# explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=cancer.feature_names, 
#                                                    categorical_features=categorical_features, 
#                                                    class_names=cancer.target_names,
#                                                    verbose=False, mode='classification',
#                                                    discretize_continuous=False,
#                                                    feature_selection='none')


#generate an explanation for testing..
i = 3
#use rf.predict_proba for classfication 
exp = explainer.explain_instance(test[i], rf.predict,#num_features=13,
                                 model_regressor='Bay_non_info_prior',
                                 num_samples=100,
                                 #labels=labels_test[i],
                                 )#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior' 'BayesianRidge_inf_prior_fit_alpha'

exp.show_in_notebook(show_table=True)
fig = exp.as_pyplot_figure(label=1)




exp = explainer.explain_instance(test[i], rf.predict,#num_features=13,
                                  model_regressor='BayesianRidge_inf_prior_fit_alpha',
                                  num_samples=100,
                                  #labels=labels_test[i],
                                  top_labels=2)
#'non_Bay' 'Bay_non_info_prior' 'Bay_info_prior', 'BayesianRidge_inf_prior_fit_alpha'
  

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

exp=calculate_posteriors.get_posterior(exp,'.\data\prior_knowledge_tabular.csv' ,hyper_para_alpha=alpha_init, hyper_para_lambda=lambda_init,
                                        label=1)

#exp.show_in_notebook(show_table=True)
print(exp.as_list())
fig = exp.as_pyplot_figure(label=1)



