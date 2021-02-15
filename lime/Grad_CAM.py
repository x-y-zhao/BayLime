import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap_vis = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, heatmap_vis


# Grad-Cam as prior knowledge
def extrat_prior(img,inet_model,explanation,fname,pred_l):
    # model parameters
    last_conv_layer_name = "mixed10"
    classifier_layer_names = [
         "avg_pool",
         "predictions",
    ]
    # last_conv_layer_name = "conv5_block3_out"
    # classifier_layer_names = [
    #         "avg_pool",
    #         "probs",
    # ]
    # last_conv_layer_name = "block14_sepconv2_act"
    # classifier_layer_names = [
    #     "avg_pool",
    #     "predictions",
    # ]
    img = np.array([img])
    grad, heatmap = make_gradcam_heatmap(img, inet_model, last_conv_layer_name, classifier_layer_names)

    img = img[0] / 2 + 0.5
    img = np.uint8(255 * img)

    # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = image.array_to_img(superimposed_img)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(superimposed_img)
    ax.set_ylabel(pred_l,fontsize=20)
    fig.savefig(fname+'/Grad_CAM_exp.png',bbox_inches='tight')
    # plt.imshow(superimposed_img)
    # plt.show()

    # resize gradmap to image size
    z = img.shape[0] / grad.shape[0]
    prior = interpolation.zoom(grad, z)

    # get the prior for each segments
    seg_prior = []
    seg_n = np.max(explanation.segments) + 1

    for i in range(seg_n):
        mask = np.where(explanation.segments == i, 0, explanation.segments)
        mask = np.where(explanation.segments != i, 1, mask)
        seg_prior.append(np.ma.array(prior, mask=mask).mean())

    return seg_prior
