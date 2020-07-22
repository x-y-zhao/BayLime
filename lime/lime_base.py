"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path,BayesianRidge
from sklearn.linear_model.modified_sklearn_BayesianRidge import BayesianRidge_inf_prior,BayesianRidge_inf_prior_fit_alpha
from sklearn.utils import check_random_state
import csv

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor='non_Bay'):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()
                XZ: change default to 'non_Bay'
                'Bay_non_info_prior' uses sklearn BayesianRidge
                'Bay_info_prior' uses XZ modified sklearn BayesianRidge
                'BayesianRidge_inf_prior_fit_alpha' uses XZ modifed 'BayesianRidge_inf_prior_fit_alpha' regressor

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        
        
        if model_regressor == 'non_Bay':
            model_reg = Ridge(alpha=1,fit_intercept=True,random_state=self.random_state)
            print('using non_Bay option for model regressor')
        
        #added by XZ
        if model_regressor == 'Bay_non_info_prior':
            #all default args
            model_reg=BayesianRidge(fit_intercept=True,
                                         n_iter=1000, tol=0.0001,
                                         verbose=True,
                                         alpha_1=1e-06, alpha_2=1e-06, 
                                         lambda_1=1e-06, lambda_2=1e-06, 
                                         alpha_init=None, lambda_init=None)
            print('using Bay_non_info_prior option for model regressor')
            
        
        #added by XZ
        #XZ: read those parameters from config files
        if model_regressor == 'Bay_info_prior':
            alpha_init=1
            lambda_init=1
            with open('./configure.csv') as csv_file:
                csv_reader=csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        alpha_init=float(row[0])
                        lambda_init=float(row[1])
                    line_count=line_count+1
            print('using Bay_info_prior option for model regressor')
            model_reg=BayesianRidge_inf_prior(fit_intercept=True,n_iter=0, tol=0.0001,  
                                         alpha_init=alpha_init, lambda_init=lambda_init)
        #XZ: we set the alpha_init and lambda_init to play with different priors
        
        #added by XZ
        #XZ: read those parameters from config files
        if model_regressor == 'BayesianRidge_inf_prior_fit_alpha':
            lambda_init=1
            with open('./configure.csv') as csv_file:
                csv_reader=csv.reader(csv_file)
                line_count = 0
                for row in csv_reader:
                    if line_count == 1:
                        lambda_init=float(row[1])
                    line_count=line_count+1
            print('using Bay_info_prior_fixed_lambda_fit_alpha option for model regressor')
            model_reg=BayesianRidge_inf_prior_fit_alpha(fit_intercept=True,n_iter=1000, tol=0.0001,  
                                         lambda_init=lambda_init,verbose=True)
        #XZ: we set the alpha_init and lambda_init to play with different priors
        
        
        
        easy_model = model_reg
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)
        
        if model_regressor == 'non_Bay':
            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))
            local_std = 0
            
        
        if model_regressor == 'Bay_info_prior' or model_regressor == 'Bay_non_info_prior' or model_regressor == 'BayesianRidge_inf_prior_fit_alpha':
            print('the alpha is',easy_model.alpha_)
            print('the lambda is',easy_model.lambda_)
            print('the regulation term lambda/alpha is', easy_model.lambda_/easy_model.alpha_)
            local_pred, local_std = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1),return_std=True)
            # Added by XZ: write the posteriors into a local file...
            with open('./posterior_configure.csv','w',newline='') as result_file:
                wr = csv.writer(result_file,delimiter=',')
                wr.writerows([['alpha','lambda']])
                wr.writerows([[easy_model.alpha_,easy_model.lambda_]])
        
        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local_mean', local_pred)
            print('Prediction_local_std', local_std,)
            print('Right:', neighborhood_labels[0, label])
        if model_regressor == 'non_Bay':
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_,np.zeros(len(easy_model.coef_))),
                           key=lambda x: np.abs(x[1]),
                           reverse=True),
                    prediction_score, local_pred)
        else:
            n_=len(easy_model.coef_)
            variance=np.zeros(n_)
            i=0
            while i<n_:
                variance[i]=easy_model.sigma_[i,i]
                i=i+1
        
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_,variance),
                       key=lambda x: np.abs(x[1]),
                       reverse=True),
                    prediction_score, local_pred)
    
    
    
    
    
    
