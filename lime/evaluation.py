import numpy as np
from os.path import join
from scipy.ndimage.filters import gaussian_filter
from copy import deepcopy
import matplotlib.pyplot as plt


HW = 299 * 299 # image area
n_classes = 1000

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode):
        r"""Create deletion/insertion metric instance.

        Args:
            model (keras.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode

    def single_run(self, org_img, exp, seg, pred_n, preds_label, exp_name, fname):
        r"""Run metric on one image-saliency pair.

        Args:
            img: normalized image array.
            explanation.segments (np.ndarray): importance of each segment.
            preds_label: the decoded prediction of orginal image

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        img = deepcopy(org_img)
        # get feature importance of each segment
        salient_order = deepcopy(exp)

        n_steps = len(salient_order)
        scores = np.zeros(n_steps)

        if self.mode == 'del':
            title = 'Deletion game'
            xlabel = 'Segments of pixels deleted'
            for i in range(n_steps):
                pred = self.model.predict(np.array([img]))
                scores[i] = pred[0, pred_n]
                # delete the segment
                if exp_name == 'Grad_CAM' or exp_name == 'SHAP':
                    seg_id = salient_order[i]
                else:
                    seg_id = salient_order[i][0]
                img[seg == seg_id] = -1.0

        elif self.mode == 'ins':
            title = 'Insertion game'
            xlabel = 'Segments of pixels inserted'
            # create a blurred image
            blur_img = gaussian_filter(img / 2 + 0.5, sigma=10)
            blur_img = (blur_img - 0.5)*2


            for i in range(n_steps):
                pred = self.model.predict(np.array([blur_img]))
                scores[i] = pred[0, pred_n]
                # insert the segment
                if exp_name == 'Grad_CAM' or exp_name == 'SHAP':
                    seg_id = salient_order[i]
                else:
                    seg_id = salient_order[i][0]
                blur_img[seg == seg_id] = img[seg == seg_id]


        f = exp_name + '_' + self.mode + '.png'
        plt.figure(figsize=(6, 6))
        plt.plot(np.arange(n_steps) / n_steps, scores)
        plt.fill_between(np.arange(n_steps) / n_steps, 0, scores, alpha=0.4)
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, max(scores)+0.1)
        plt.text(0.5, (max(scores)+0.1)/2, ' AUC: ' + str(round(auc(scores),6)), ha="center", va="center", zorder=10, fontsize = 20)
        plt.title( exp_name + '_' + title)
        plt.xlabel(xlabel)
        plt.ylabel(preds_label[0][1])
        plt.savefig(join(fname, f),bbox_inches='tight')


        return auc(scores)

