import sys
import inspect
import types
import numpy as np
import math
from scipy.spatial import distance


def cal_iou(exp_mask,org_mask):
    union = np.logical_or(exp_mask, org_mask)
    inter = np.logical_and(exp_mask, org_mask)
    exp_iou = np.count_nonzero(inter) * 1. / np.count_nonzero(union)
    return exp_iou

def cal_dist(exp_mask, org_mask):
    ll = np.argwhere(exp_mask)
    ll2 = np.argwhere(org_mask)
    min_array = distance.cdist(ll, ll2).min(axis=1)
    scale_len = math.sqrt((exp_mask.shape[0] - 1) ** 2 + (exp_mask.shape[1] - 1) ** 2)
    min_distance = sum(min_array) / (len(min_array) * scale_len)
    return min_distance





def has_arg(fn, arg_name):
    """Checks if a callable accepts a given keyword argument.

    Args:
        fn: callable to inspect
        arg_name: string, keyword argument name to check

    Returns:
        bool, whether `fn` accepts a `arg_name` keyword argument.
    """
    if sys.version_info < (3,):
        if isinstance(fn, types.FunctionType) or isinstance(fn, types.MethodType):
            arg_spec = inspect.getargspec(fn)
        else:
            try:
                arg_spec = inspect.getargspec(fn.__call__)
            except AttributeError:
                return False
        return (arg_name in arg_spec.args)
    elif sys.version_info < (3, 6):
        arg_spec = inspect.getfullargspec(fn)
        return (arg_name in arg_spec.args or
                arg_name in arg_spec.kwonlyargs)
    else:
        try:
            signature = inspect.signature(fn)
        except ValueError:
            # handling Cython
            signature = inspect.signature(fn.__call__)
        parameter = signature.parameters.get(arg_name)
        if parameter is None:
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))
