import matplotlib
import matplotlib.cm

import tensorflow as tf
import numpy as np
# Taken from: https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'hot')

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    value = tf.cast(value,tf.float32)
    vmin = tf.reduce_min(input_tensor=value) if vmin is None else vmin
    vmax = tf.reduce_max(input_tensor=value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax
    value = tf.clip_by_value(value, 0,1)
    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.cast(tf.round(value * 255), dtype=tf.int32)
    # indices = tf.cast(value * 255.0, dtype=tf.uint8)

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'hot')
    colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    # colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def colorize1(value, vmin=None, vmax=None, colors=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'hot')

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```

    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(input_tensor=value) if vmin is None else vmin
    vmax = tf.reduce_max(input_tensor=value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.cast(tf.round(value * 255), dtype=tf.int32)
    # indices = tf.cast(value * 255.0, dtype=tf.uint8)

    # gather
    # cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'hot')
    # colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    # colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


# def decode_labels_reg(mask, num_images=1):
#     """Decode batch of segmentation masks.
#
#     Args:
#       mask: result of regression inference.
#       num_images: number of images to decode from the batch.
#
#     Returns:
#       A batch with num_images RGB images of the same size as the input.
#     """
#
#     img = colorize(mask[i],vmin=0,vmax=4,cmap='hot')
#         # img = plot_heatmap(mask[i],min=0,max=4)
#         # outputs[i] = np.array(img)[:, :, 0:3]
#         outputs.append(img)
#     return outputs

def inv_preprocess_tf(imgs, img_mean, scale_luminosity, s2=True):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      scale_luminosity: scale used to feed the images to the NN.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """

    # value = (imgs + img_mean) * scale_luminosity
    value = (imgs * scale_luminosity) + img_mean

    # img = (imgs[i]) * scale_luminosity
    # img = scale * imgs[i] / (1 - imgs[i])
    if s2:
        value = plot_rgb(value)
    else:
        value = tf.cast(value * 255.0, dtype=tf.uint8)
    return value



def plot_rgb(value, max_luminance=4000, percentiles = (1,99), reorder = True):

    mi = tf.contrib.distributions.percentile(value,q=percentiles[0])
    ma = tf.contrib.distributions.percentile(value, q=percentiles[1])

    # if mi < max_luminance:
    ma = tf.minimum(ma, max_luminance)

    # normalize
    mi = tf.reduce_min(input_tensor=value) if mi is None else mi
    ma = tf.reduce_max(input_tensor=value) if ma is None else ma
    value = (value - mi) / (ma - mi)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    # value = tf.to_int32(tf.round(value * 255))
    value = tf.cast(value * 255.0, dtype=tf.uint8)
    if reorder:
        return slice_last_dim(value, (2,1,0))
    else:
        return value

def slice_last_dim(tensor,dims):
    output = None
    exp_dim = len(tensor.get_shape().as_list()) -1

    for val in dims:
        tensor_ = tensor[...,val]
        tensor_ = tf.expand_dims(tensor_,axis=exp_dim)
        output = tf.concat([output,tensor_],axis=exp_dim) if output is not None else tensor_
    return output


def plot_heatmap(data, min=None, max=None, percentiles=(1,99), cmap = 'hot'):

    data[np.isnan(data)] = -1

    mi, ma = np.nanpercentile(data, percentiles)
    if min is None:
        min = mi
    if max is None:
        max = ma

    band_data = np.clip(data, min, max)
    if min < max: # added for when all uniform pixels are given
        band_data = (band_data - min) / (max - min)

    band_data  = np.squeeze(band_data)

    assert len(band_data.shape) == 2, 'For more channels in image use plot_rgb'

    cm = plt.get_cmap(cmap)
    return Image.fromarray(cm(band_data, bytes=True))