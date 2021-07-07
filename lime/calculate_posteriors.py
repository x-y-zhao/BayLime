# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:01:26 2020

@author: Xingyu Zhao
"""

import pandas as pd
import numpy as np

# this function will be relocated to somefile... 
def get_posterior(exp,prior_knowledge, hyper_para_alpha, hyper_para_lambda,label=1):
    '''
    Added by XZ to calculate posteriors when mu0 is non zero
    Parameters
    ----------
    exp : a lime explaination instance
    hyper_para_alpha : the alpha used in the BayesianLinearRegression
    hyper_para_lambda : the lambda used in the BayesianLinearRegression
    label: which label to explain..
    Returns
    -------
    A lime explain_instance type (a dictionary)

    '''


    feature_name_list = exp.as_list(label=label)
        
    feature_id_list = [(x[0], float(x[1]),float(x[2])) for x in exp.local_exp[label]]
    
    temp_list=[feature_id_list[i]+feature_name_list[i] for i in range(0, len(feature_id_list))]

    # max-abs scale prior knowledge
    ob = [float(x[1]) for x in exp.local_exp[label]]
    ob_abs_max = max(abs(np.array(ob)))
    pr_abs_max = max(abs(np.array(prior_knowledge)))
    pr = prior_knowledge/pr_abs_max*ob_abs_max

    # min-max scale
    # ob_max = max(ob)
    # ob_min = min(ob)
    # pr_max = max(prior_knowledge)
    # pr_min = min(prior_knowledge)
    # pr = (prior_knowledge-pr_min)/(pr_max-pr_min)*(ob_max-ob_min)+ob_min

    
    # two new lists of tuple
    #new_list_with_feature_names=[]
    new_list_with_feature_index=[]
    #print((temp_list))
    for x in temp_list:
        if x[0]>=len(pr):#this is the case that our prior knowldege didn't cover all features..
            t_=(hyper_para_lambda*x[2])*0+x[1]
            new_list_with_feature_index.append((x[0],t_,x[2]))
        else:
            t_=(hyper_para_lambda*x[2])*pr[x[0]]+x[1]
            new_list_with_feature_index.append((x[0],t_,x[2]))
    
    # for index, x in enumerate(new_list_with_feature_names):
    #     new_list_with_feature_index.append((exp.local_exp[label][index][0],x[1],x[2]))
    
    
    #print(new_list_with_feature_names)
    dict_={label:sorted(new_list_with_feature_index,key=lambda x: np.abs(x[1]),reverse=True)}
    exp.local_exp.update(dict_)
    return exp

