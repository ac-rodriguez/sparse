import pprint
import os
import string
import time
import numpy as np
from math import ceil
from plots import plot_rgb, plot_heatmap
import matplotlib.pyplot as plt

def save_parameters(params, out_dir, sysargv=None, name='FLAGS'):

    params = params.__dict__

    with open(os.path.join(out_dir, name + '.txt'), 'w') as f:
        sorted_names = sorted(params.keys(), key=lambda x: x.lower())
        for key in sorted_names:
            value = params[key]
            f.write('%s:%s\n' % (key, value))

        if sysargv:
            f.write("\n\n")
            for i in sysargv:
                f.write("{} ".format(i))

    pprint.pprint(params)

    print(' '.join(sysargv))

def add_letter_path(out_dir, timestamp = True):
    ''' Add a letter to the output directory of the summaries to avoid overwriting if several jobs are run at the same time'''

    if timestamp:
        out_dir = out_dir + time.strftime("%y%m%d_%H%M", time.gmtime())
    i = 0
    letter = ''
    created = False

    while not created:
        if not os.path.exists(out_dir + letter):
            try:
                os.makedirs(out_dir + letter)
                created = True
            except:
                pass
        # Check if the folder contains any kind of file
        elif len([name for name in os.listdir(out_dir + letter) if os.path.isfile(os.path.join(out_dir + letter, name))]) == 0:
            created = True
        else:
            letter = string.ascii_lowercase[i]
            i += 1
    return out_dir + letter




def decode_labels_reg(mask, num_images=1):
    """Decode batch of segmentation masks.

    Args:
      mask: result of regression inference.
      num_images: number of images to decode from the batch.

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = plot_heatmap(mask[i],min=0,max=4)
        outputs[i] = np.array(img)[:, :, 0:3]
    return outputs

def inv_preprocess(imgs, num_images, img_mean, scale_luminosity, reorder=True):
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
    n, h, w, c = imgs.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = (imgs[i] + img_mean) * scale_luminosity
        # img = (imgs[i]) * scale_luminosity
        # img = scale * imgs[i] / (1 - imgs[i])
        img = plot_rgb(img[...,0:3], file='', return_img=True, reorder=reorder)
        outputs[i] = np.array(img)
    return outputs


def recompose_images(a, border= 4, size=None, show_image=False):

    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2]-border*2

        print('Patch has dimension {}'.format(patch_size))
        print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[0]/float(patch_size)))
        y_tiles = int(ceil(size[1]/float(patch_size)))
        print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        print('Image size is: {}'.format(size))
        images = np.zeros((size[1], size[0],a.shape[3])).astype(np.float32)

        print(images.shape)
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[1] - patch_size:
                ypoint = size[1] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[0] - patch_size:
                    xpoint = size[0] - patch_size
                images[ypoint:ypoint+patch_size, xpoint:xpoint+patch_size,:] = a[current_patch, border:a.shape[2]-border, border:a.shape[2]-border,:]
                current_patch += 1
    # Debug images
    #         if show_image:
    #             nice_imshow(images[1, :, :])
    #             plt.show()
    return images
    # return images.transpose((1, 2, 0))
