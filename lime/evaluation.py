import numpy as np
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

    def single_run(self, org_img, explanation, preds_label):
        r"""Run metric on one image-saliency pair.

        Args:
            img: normalized image array.
            explanation.segments (np.ndarray): importance of each segment.
            preds_label: the decoded prediction of orginal image

        Return:
            scores (nd.array): Array containing scores at every step.
        """
        img = deepcopy(org_img)
        pred_label = explanation.top_labels[0]
        # get feature importance of each segment
        salient_order = deepcopy(explanation.local_exp[pred_label])

        n_steps = len(salient_order)
        scores = np.zeros(n_steps)

        if self.mode == 'del':
            title = 'Deletion game'
            xlabel = 'Segments of pixels deleted'
            for i in range(n_steps):
                pred = self.model.predict(np.array([img]))
                scores[i] = pred[0, pred_label]
                # delete the segment
                seg_id = salient_order[i][0]
                img[explanation.segments == seg_id] = 0

        elif self.mode == 'ins':
            title = 'Insertion game'
            xlabel = 'Segments of pixels inserted'
            # create a blurred image
            blur_img = gaussian_filter(img / 2 + 0.5, sigma=5)
            blur_img = (blur_img - 0.5)*2

            plt.imshow(blur_img / 2 + 0.5)
            plt.show()

            for i in range(n_steps):
                pred = self.model.predict(np.array([blur_img]))
                scores[i] = pred[0, pred_label]
                # insert the segment
                seg_id = salient_order[i][0]
                blur_img[explanation.segments == seg_id] = img[explanation.segments == seg_id]

            plt.imshow(blur_img / 2 + 0.5)
            plt.show()


        plt.figure(figsize=(5, 5))
        plt.plot(np.arange(n_steps) / n_steps, scores)
        plt.fill_between(np.arange(n_steps) / n_steps, 0, scores, alpha=0.4)
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, max(scores)+0.1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(preds_label[0][1])
        plt.show()


        return auc(scores)

