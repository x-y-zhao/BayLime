# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:01:26 2020

@author: xz45
"""

import pandas as pd
import numpy as np

# this function will be relocated to somefile... 
def get_posterior(exp, hyper_para_alpha, hyper_para_lambda,num_samples,label=1):
    '''
    Added by XZ to calculate posteriors when mu0 is non zero
    Parameters
    ----------
    exp : a lime explaination instance
    hyper_para_alpha : the alpha used in the BayesianLinearRegression
    hyper_para_lambda : the lambda used in the BayesianLinearRegression
    num_samples : number of data points used in the BayesianLinearRegression.
    label: which label to explain..
    Returns
    -------
    A lime explain_instance type (a dictionary)

    '''
    #read from the prior knowledge repo
    w=pd.read_csv(r'.\data\prior_knowledge.csv')
    temp_list=exp.as_list(label=label)
    
    # a new list of tuple
    new_list_with_feature_names=[]
    new_list_with_feature_index=[]
    #print((temp_list))
    for x in temp_list:
        for col in w.columns:
            if x[0]==col:
                t_=(hyper_para_lambda/(hyper_para_lambda+hyper_para_alpha*num_samples))*w[col].mean()+x[1]
                new_list_with_feature_names.append((col,t_))
    
    for index, x in enumerate(new_list_with_feature_names):
        new_list_with_feature_index.append((exp.local_exp[label][index][0],x[1]))
    
    
    #print(new_list_with_feature_names)
    dict_={label:sorted(new_list_with_feature_index,key=lambda x: np.abs(x[1]),reverse=True)}
    exp.local_exp.update(dict_)
    return exp

