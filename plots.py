import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from matplotlib.colors import LinearSegmentedColormap


def check_dims(data):
    data = np.squeeze(data)
    ndims = len(data.shape)
    if ndims == 3:
        min_dim = np.min(data.shape)
        if data.shape.index(min_dim) == 0:
            data = np.expand_dims(data, axis=3)
        else:
            data = np.expand_dims(data, axis=0)
    elif ndims == 2:
        data = np.expand_dims(data, axis=3)
        data = np.expand_dims(data, axis=0)
    return data

def plot_heatmap(data, min=None, max=None, percentiles=(1,99), cmap='viridis', file=None):

    data[np.isnan(data)] = -1
    data = np.float32(data)
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
    if file is None:
        return Image.fromarray(cm(band_data, bytes=True))
    else:
        Image.fromarray(cm(band_data, bytes=True)).save(file+'.png')

def plot_rgb(data, file, max_luminance=4000, reorder=True, return_img=False, percentiles = (1,99), bw = False, normalize=True):

    data[np.isnan(data)] = -1

    if normalize:
        mi, ma = np.nanpercentile(data, percentiles)
        if mi < max_luminance:
            ma = min(ma, max_luminance)

        band_data = np.clip(data, mi, ma)
        if mi < ma: # added for when all uniform pixels are given
            band_data = (band_data - mi) / (ma - mi)
    else:
        band_data = data
    band_data  = np.squeeze(band_data)

    if len(band_data.shape) == 2: ## If only one band is present plot as a heatmap
        cm = plt.get_cmap('jet')
        img = Image.fromarray(cm(band_data, bytes=True))
    else:
        if reorder:
            band_data = band_data[:, :, (2, 1, 0)]

        if bw:
            band_data = np.dot(band_data[...,:3], [0.299, 0.587, 0.114])
            band_data = np.stack((band_data,) * 3, -1)
        img = Image.fromarray(np.uint8(band_data*255))
        # converter = ImageEnhance.Color(img)
        # img = converter.enhance(1.5)

    if return_img:
        return img
    else:
        img.save(file + '.png')
        print('{} saved'.format(file + '.png'))





def plot_labels(preds, file, colors=None, return_img=False):

    filename = file + '.png'
    preds  = np.squeeze(preds)
    if colors is not None:
        colors1 = [tuple([y / 255.0 for y in x]) for x in colors]
        cm = LinearSegmentedColormap.from_list('my_cmap', colors1, N=len(colors))
    else:
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        cm = plt.get_cmap("tab20")

    img = Image.fromarray(cm(preds, bytes=True))
    if return_img:
        return img
    else:
        img.save(filename)
        print('{} saved'.format(filename))

def plot_rgb_mask(data, preds, file, colors, max_luminance=4000, reorder=True, omit_id_clouds = 0):

    img_labels = plot_labels(preds=preds,file = '', colors = colors, return_img=True)

    imgRGB = plot_rgb(data,file = '', max_luminance=max_luminance, reorder=reorder,return_img=True)
    converter = ImageEnhance.Color(imgRGB)
    imgRGB = converter.enhance(2.0)
    alpha = 0.3
    img_alpha = Image.fromarray(np.uint8(alpha* 255 * (preds.squeeze()>omit_id_clouds))) ## put full transparency to class 0 (background no clouds)
    img_labels.putalpha(img_alpha)

    imgRGB = imgRGB.copy()
    imgRGB.paste(img_labels,None,img_labels)

    filename = file + '.png'
    imgRGB.save(filename)
    print('{} saved'.format(filename))

def plot_rgb_density(data, preds, file, max_luminance=4000, reorder=True, omit_id_clouds = 0.5, pred_range = [0,2.5], bw = False, cmap = 'hot', cld = None, colors = None):


    img_labels = plot_heatmap(preds, min = pred_range[0], max = pred_range[1], cmap=cmap)

    imgRGB = plot_rgb(data,file = '', max_luminance=max_luminance, reorder=reorder,return_img=True, bw = bw)
    # converter = ImageEnhance.Color(imgRGB)
    # imgRGB = converter.enhance(1.5)
    alpha = 0.5
    img_alpha = Image.fromarray(np.uint8(alpha* 255 * (preds.squeeze()>omit_id_clouds))) ## put full transparency to class 0 (background no clouds)
    img_labels.putalpha(img_alpha)

    imgRGB = imgRGB.copy()
    imgRGB.paste(img_labels,None,img_labels)

    # if cld is not None:
    #     img_cld = plot_labels(preds=cld, file='', colors=colors, return_img=True)
    #
    #     img_alpha1 = Image.fromarray(np.uint8(alpha * 255 * (
    #     cld.squeeze() > omit_id_clouds)))  ## put full transparency to class 0 (background no clouds)
    #     img_cld.putalpha(img_alpha1)
    #
    #     imgRGB = imgRGB.copy()
    #     imgRGB.paste(img_cld, None, img_cld)

    filename = file + '.png'
    imgRGB.save(filename)
    print('{} saved'.format(filename))

def cmap_discretize(n_colors, label_colours):
    """Return a discrete colormap from the continuous colormap cmap.

    Parameters
    ----------
    cmap : str or colormap object
        Colormap to discretize.
    n_colors : int
        Number of discrete colors to divide `cmap` into.

    Returns
    ----------
    discrete_cmap : LinearSegmentedColormap
        Discretized colormap.
    """
    label_colours.append((0., 0., 0.))
    colors_rgba = np.array([x + (1,)for x in label_colours])
    indices = np.linspace(0, 1., n_colors + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(n_colors + 1)]
    # Return colormap object.
    return LinearSegmentedColormap('custom_cmap_{}'.format(n_colors), cdict, 1024)