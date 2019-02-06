"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import glob
import pickle
import csv
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np
from sklearn import metrics
import shutil

# from deeplab_resnet.model import DeepLabResNetModel
# from deeplab_resnet import model as models_resnet
# from deeplab_resnet.utils import decode_labels, prepare_label, save_parameters
# from color_codes import EncodeLabels
# from read_sentinel import plots, patches
import plots
from model import model_fn
from utils import save_parameters, add_letter_path
from data_reader import read_and_upsample_sen2, read_labels
from color_codes import EncodeLabels


NUM_CLASSES = 3
SAVE_DIR = '/scratch/andresro/palm/evaluation/coco_sparse'
POINTSFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_detections.kml'
LRFILE = "/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml"

Encoder = EncodeLabels()

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    # parser.add_argument("img_path", type=str,
    #                     help="Path to the RGB image file.")
    parser.add_argument("model_dir", type=str,
                        help="Path to the file with model weights.")
    parser.add_argument("--HR_file", default=None)
    parser.add_argument("--LR_file", default=LRFILE)
    parser.add_argument("--points", default=POINTSFILE)
    parser.add_argument("--roi_lon_lat_val", default='117.84,8.82,117.92,8.9')
    parser.add_argument("--roi_lon_lat_val_lb", default='117.8821,8.87414,117.891,8.8654')
    # parser.add_argument("--roi_lon_lat_val", default='117.81,8.82,117.84,8.88')
    # parser.add_argument("--roi_lon_lat_val_lb", default='117.820,8.848,117.834,8.854')
    # parser.add_argument("--roi_lon_lat_val_lb", default='117.81,8.82,117.84,8.88')
    parser.add_argument("--select_bands", default="B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12",
                        help="Select the bands. Using comma-separated band names.")

    parser.add_argument("--model", type=str, default='DL2',
                        help="Model to be used for training [DL2 (DeepLab-V2), Simple(Only ResNet Blocks), DL3].")
    parser.add_argument("--img-size", type=str, default='200,200',
                        help="size of the input image to be chosen")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    parser.add_argument("--xy-corner", type=str, default="0,0",
                        help="Where to save predicted mask.")
    parser.add_argument("--pred-range", type=str, default="0,2",
                        help="Prediction Range for plotting.")
    parser.add_argument("--no-cloud-predictor", dest='use_cloud_predictor',default=True, action="store_false", help="Use cloud mask as a predictor.")
    parser.add_argument('--tag', dest='tag',
                        help='tag for saving', default=None,
                        nargs=argparse.REMAINDER)
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''

    if '.ckpt' not in ckpt_path:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = ckpt.model_checkpoint_path
            counter = int(ckpt_path.split('-')[-1])
        else:
            print("Invalid ckpt_path {}".format(ckpt_path))
            sys.exit(0)

    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def chi2_distance(peaksA, peaksB, eps=1e-10, **kwargs):
    histA, _ = np.histogram(peaksA, **kwargs)
    histB, _ = np.histogram(peaksB, **kwargs)

    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d

def compute_metrics(pred, label, pred_reg, label_reg, cld = None, id_file = None):



    _units = (1 / 100.00)**2 # from 10m pixels to km2

    metrics1 = {}

    metrics1['Filename'] = id_file
    pred_reg = pred_reg.squeeze()
    label_reg = label_reg.squeeze()

    pred = pred.flatten()
    label = label.flatten()
    cld = cld.flatten()
    total_area = pred.shape[0]*_units

    metrics1['Area(Km^2)'] = total_area

    cld_enc = Encoder.encode(np.zeros_like(cld),cld)
    for i in range(3):
        metrics1['%Cl{}'.format(i)] = np.mean(cld_enc == i)


    # for i in range(3):
    #     pr_ = np.sum(pred == i) *_units
    #     lb_ = np.sum(label == i) *_units
    #
    #     metrics1['GT_{}'.format(Encoder.names[i])] = lb_
    #     metrics1['Pred_{}'.format(Encoder.names[i])] = pr_
    # for i in range(3):
    #     weight = cld_enc == i
    #     if np.all(weight == 0):
    #         acc_cl = prec_cl = recall_cl = None
    #     else:
    #         acc_cl = metrics.accuracy_score(label == 0, pred == 0, sample_weight = weight)
    #         prec_cl = metrics.precision_score(label == 0, pred == 0, sample_weight = weight)
    #         recall_cl = metrics.recall_score(label == 0, pred == 0, sample_weight = weight)
    #
    #     metrics1['Acc_Cl{}'.format(i)] = acc_cl
    #     metrics1['Prec_Cl{}'.format(i)] = prec_cl
    #     metrics1['Recall_Cl{}'.format(i)] = recall_cl

    pred_r_int = np.round(pred_reg)
    label_r_int = np.round(label_reg)

    metrics1['Obj_GT'] = np.sum(label_reg)
    metrics1['Obj_Pred'] = np.sum(pred_reg)
    metrics1['Obj_GT(Rounded)'] = np.sum(label_r_int)
    metrics1['Obj_Pred(Rounded)'] = np.sum(pred_r_int)
    metrics1['Obj_diff%(Rounded)'] = np.sum(pred_r_int)/np.sum(label_r_int) - 1
    metrics1['Obj_diff%'] = np.sum(pred_reg) / np.sum(label_reg) - 1

    chi2_d = chi2_distance(pred_reg,label_reg)
    metrics1['Chi2 Distance'] = chi2_d



    TP = np.sum(np.logical_and(label == 1, pred == 1))
    TN = np.sum(np.logical_and(label == 0, pred == 0))
    FP = np.sum(np.logical_and(label == 0, pred == 1))
    FN = np.sum(np.logical_and(label == 1, pred == 0))

    iou = TP / float(TP + FP + FN)
    acc = (TP + TN) /float(TP + FP + FN + TN)
    prec = TP / float(TP + FP)
    recall = TP / float(TP + FN)

    mean_abs_err = np.mean(np.abs(pred_reg - label_reg))
    mean_sq_err = np.mean(np.square(pred_reg - label_reg))

    metrics1['MeanAbsErr'] = mean_abs_err
    metrics1['MeanSqErr'] = mean_sq_err
    metrics1['IoU'] = iou
    metrics1['Acc'] = acc
    metrics1['Precision'] = prec
    metrics1['Recall'] = recall


    metrics1['summary_metrics1'] = ' {} & {} & {} && {} & {} & {}'.format(iou, prec, recall, mean_sq_err, mean_abs_err , chi2_d)
    print(' {} & {} & {} && {} & {} & {}'.format(iou, prec, recall, mean_sq_err, mean_abs_err , chi2_d))

    metrics1['summary_objects'] = ' {} & {} & {} '.format(np.sum(label_reg), np.sum(pred_reg), np.sum(pred_reg) / np.sum(label_reg) - 1)
    print(' {} & {} & {} '.format(np.sum(label_reg), np.sum(pred_reg), np.sum(pred_reg) / np.sum(label_reg) - 1))

    for key in metrics1:
        metrics1[key] = [metrics1[key]]

    return metrics1

def merge_metrics(metricsA, metricsB):
    if metricsB:
        metrics_merged = {}
        for key in set().union(metricsA, metricsB):
            if key in metricsA: metrics_merged.setdefault(key, []).extend(metricsA[key])
            if key in metricsB: metrics_merged.setdefault(key, []).extend(metricsB[key])
    else:
        metrics_merged = metricsA

    return metrics_merged

def write_metrics(pred, label, pred_reg, label_reg, cld = None, file='metrics', id_file = None):

    # if not os.path.exists(file):
    #     f = open(file, "w")
    #
    #     headers = ['Filename','Area(Km^2)','%Cl0','%Cl1','%Cl2']
    #     # f.write("FileName\tArea(Km^2)\t")
    #
    #     for i in range(3):
    #         headers.extend(['GT_{}'.format(Encoder.names[i]),'Pred_{}'.format(Encoder.names[i])])
    #
    #     headers.extend(['IoU','Acc','Precision','Recall'])
    #
    #     headers.extend(['GT_Coco','Pred_Coco','GT_Coco_round','Pred_Coco_round'])
    #
    #     headers.extend(['Chi2_distance', 'MeanAbsError','MeanSquaredError'])
    #
    #     f.write('\t'.join(headers))
    #     f.write('\n')
    #
    #     f.close()
    if not file.endswith('.txt'):
        file = file + '.txt'
    f = open(file,"w")
    f.write('File:{}\n'.format(id_file))

    _units = (1 / 100.00)**2 # from 10m pixels to km2

    pred = pred.flatten()

    label = label.flatten()
    cld = cld.flatten()
    total_area = pred.shape[0]*_units
    f.write('Total Area = {:.2f}km^2\n'.format(total_area))

    cld_enc = Encoder.encode(np.zeros_like(cld),cld)
    f.write('Cloud Coverage:\n')
    for i in range(3):
        f.write('Clouds {}: {:.1f}%\n'.format(i,100.0*np.mean(cld_enc == i)))
    f.write('Prediction per Class:\n')

    for i in range(3):

        pr_ = np.sum(pred == i) *_units
        lb_ = np.sum(label == i) *_units
        f.write('{} GT:{:.3f}\tPred:{:.3f}\n'.format(Encoder.names[i],pr_,lb_))


    iou = metrics.jaccard_similarity_score(label,pred)
    acc = metrics.accuracy_score(label == 0, pred == 0)
    f.write('IoU: {:.3f}\tAcc:{:.3f}\n'.format(iou, acc))



    prec = metrics.precision_score(label == 0, pred==0)
    recall = metrics.recall_score(label == 0, pred==0)
    # f.write('Precision{:.3f}\t{:.3f}\t'.format(prec, recall))

    f.write('Precision={:.3f},\tRecall={:.3f}\n'.format(prec, recall))

    for i in range(3):
        weight = cld_enc == i
        if np.all(weight == 0):
            acc = prec = recall = -1
        else:
            acc = metrics.accuracy_score(label == 0, pred == 0, sample_weight = weight)
            prec = metrics.precision_score(label == 0, pred == 0, sample_weight = weight)
            recall = metrics.recall_score(label == 0, pred == 0, sample_weight = weight)

        f.write('Cl{} Acc={:.3f}\t'.format(i, acc))
        f.write('Prec={:.3f}\t'.format(prec))
        f.write('Rec={:.3f}\n'.format(recall))


#    f.write("\n\n")
    f.write('Density Estimation Coconut:\n')

    pred_reg = np.clip(pred_reg, 0, 4)

    pred_r_int = np.round(pred_reg)
    label_r_int = np.round(label_reg)

    # f.write('{:.1f}\t{:.1f}\t'.format(np.sum(label_reg), np.sum(pred_reg)))
    # f.write('{}\t{}\t'.format(np.sum(label_r_int), np.sum(pred_r_int)))
    p,l = np.sum(pred_r_int), np.sum(label_r_int)
    f.write('Tree Count (rounded): GT={},\tPred={}\tDiff={}\n'.format(p,l,abs(p-l)))
    p, l = np.sum(pred_reg), np.sum(label_reg)
    f.write('Tree Count: GT={:.1f},\tPred={:.1f}\n'.format(p,l,abs(p-l)))

#     chi2_d = chi2_distance(pred_reg,label_reg)
#     f.write('{:.3f}\t'.format(chi2_d))
#
#     mean_abs_err = metrics.mean_absolute_error(pred_reg.flatten(),label_reg.flatten())
#     mean_sq_err = metrics.mean_squared_error(pred_reg.flatten(),label_reg.flatten())
#
# #    f.write('Mean Abs Error={:.3f},\tMean Sq Error={:.3f}\n'.format(mean_abs_err, mean_sq_err))
#     f.write('{:.3f}\t{:.3f}\t'.format(mean_abs_err, mean_sq_err))
#     f.write('\n')
    f.close()

    print('Metrics writen to {}'.format(file))

def write_to_file(mydict, filename):

    if not filename.endswith('.csv'):
        filename = filename +'.csv'

    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sorted(mydict.items()):
            writer.writerow([key]+value)

def plot_reg(data, file, min = 0, max = 4):
    if not file.endswith('.png'):
        file = file + '.png'
    img = plots.plot_heatmap(data, min=min, max=max)
    img.save(file)
    print('{} saved'.format(file))

def plot_diff(data, file):
    if not file.endswith('.png'):
        file = file + '.png'
    img = plots.plot_heatmap(data, percentiles=(0,100), cmap='cool')
    img.save(file)
    print('{} saved'.format(file))


def plot_histograms(real, fake, filename = '', predrange = [0,2]):
    n_bins = 10
    chi2 = chi2_distance(real, fake, range = predrange, bins = n_bins)

    bins = np.linspace(predrange[0], predrange[1], n_bins, dtype=np.float64)

    real_hist, _ = np.histogram(real,density=True,range=predrange,  bins= n_bins)
    fake_hist, _ = np.histogram(fake, range=predrange, density=True, bins=n_bins)

    plt.plot(bins,real_hist / np.sum(real_hist), label = 'GT')
    plt.plot(bins, fake_hist / np.sum(fake_hist), label = 'Pred')
    plt.ylim(0,0.4)
    plt.title('Histogram \n $\chi^2 Distance = {:.2}$'.format(chi2))

    plt.legend()
    plt.savefig(filename+'.png')
    plt.close()

def plot_inference(label, pred, name):
    min = 0
    max = 4
    plt.axis('off')
    f, axarr = plt.subplots(1, 2)

    color = 'black'
    cmap = ['hot', 'cool']
    im = {}
    data = [pred, (label - pred)]
    for j, ax in enumerate(axarr):
        size = label.shape[0]
        ax.set_aspect('equal')
        plt.axis([0, size, 0, size])
        ax.tick_params(axis='both',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       left='off', right='off', labelbottom='off',
                       labelleft='off')  # labels along the bottom edge are off
        try:
            [i.set_color(color) for i in ax.spines.itervalues()]
            [i.set_linewidth(2) for i in ax.spines.itervalues()]
        except:
            [i.set_color(color) for i in ax.spines.values()]
            [i.set_linewidth(2) for i in ax.spines.values()]
        im[j] = ax.pcolormesh(data[j], cmap=cmap[j], vmin=min, vmax=max, edgecolors='face')
        f.colorbar(im[j], ax=ax)
    f.savefig(name + '.png', dpi=f.dpi)
    plt.close(f)

def plot_colorbar(save_dir, pred_range):
    import matplotlib as mpl

    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=(8, 2))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=pred_range[0], vmax=pred_range[1])

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Object Density')
    file = 'density_bar_{}-{}'.format(pred_range[0], pred_range[1])
    file = file.replace('.', '_')
    filename = os.path.join(save_dir,file)

    plt.savefig(filename+'.png', bbox_inches='tight', dpi=196)  # , bbox_extra_artists=(txt_top)) #, txt_left))  # Save Image

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    # save_dir = args.save_dir
    # n_channels = 11
    # ix, iy = [int(x) for x in args.xy_corner.split(",")]
    # pred_range = [float(x) for x in args.pred_range.split(",")]


    # data_name = ''.format(os.path.basename(args.HR_file).replace('.tif',''))
    # model_dir = os.path.join(args.save_dir,'snapshots','model-{}_size-{}_scale-{}_nchan{}{}'.format(args.model,args.patch_size, args.scale,args.n_channels,args.tag))
    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        print(' Choose an existing trained model directory')
        sys.exit(1)



    save_parameters(args,model_dir, sys.argv, name='FLAGS_eval')
    params = {}



    params['model_dir'] = model_dir
    params['args'] = args


    run_config = tf.estimator.RunConfig()

    model = tf.estimator.Estimator(model_fn=partial(model_fn,params=params),
                                   model_dir=model_dir, config=run_config)

    tf.logging.set_verbosity(tf.logging.INFO)# Show training logs.

    data = read_and_upsample_sen2(args, roi_lon_lat=args.roi_lon_lat_val)
    points = read_labels(args, roi=args.roi_lon_lat_val, roi1=args.roi_lon_lat_val).squeeze().astype(np.float32)


    # args = get_arguments()
    # n_channels = 11
    ix, iy = [int(x) for x in args.xy_corner.split(",")]
    pred_range = [float(x) for x in args.pred_range.split(",")]

    size = [int(x) for x in args.img_size.split(",")]

    data = data[ix:(ix + size[0]), iy:(iy + size[1]), :]

    points = points[ix:(ix + size[0]), iy:(iy + size[1])]


    input_fn = lambda : tf.data.Dataset.from_tensors(np.expand_dims(data,0))

    preds_gen = model.predict(input_fn=input_fn)  # ,predict_keys=['hr_hat_rgb'])
    p_ = preds_gen.next()
    pred_regs = p_['y_hat'].squeeze()

    preds =np.int8(pred_regs > 1)
    label = np.int8(points > 1)

    # reader = DataReader(args,is_training=False)




    # # def load_image(path, ix, iy, size):
    # #     # Prepare image.
    # #     with np.load(path) as data_:
    # #         data10 = data_['data10']
    # #         data20 = data_['data20']
    # #         cld = data_['cld20']
    # #         label = data_['label10']
    # #         points = np.float32(data_['points'])
    # #     if data10.shape[0:2] != data20.shape[0:2]:
    # #         data20 = patches.interpPatches(data20, data10.shape[0:2], squeeze=True)
    # #         cld = patches.interpPatches(cld, data10.shape[0:2], squeeze=True)
    # #     img_orig = np.concatenate((data10, data20, cld), axis=2)
    # #     img_orig = img_orig[ix:(ix+size[0]),iy:(iy+size[1]),:]
    # #     label = label[ix:(ix + size[0]), iy:(iy + size[1])]
    # #     points = points[ix:(ix + size[0]), iy:(iy + size[1])]
    # #     return img_orig, label, points
    # #
    # # data_placeholder = tf.placeholder(tf.float32, shape=[size[0], size[1], n_channels])
    # # IMG_MEAN = tf.Variable(np.zeros(n_channels), name='mean_train', trainable= False, dtype=tf.float32)
    # # SCALE = tf.Variable(9999.0, name='scale_preprocessing', trainable = False, dtype=tf.float32)
    # #
    # # data = data_placeholder / SCALE
    # # data = data - IMG_MEAN
    # #
    # # if not args.use_cloud_predictor:
    # #     image_batch1 = data[:, :, :, :-1]
    # #
    # # else:
    # #     image_batch1 = data
    # #
    # # model, n_layers = models_resnet.get_model(args.model)
    # #
    # # # Create network.
    # # net = model({'data': tf.expand_dims(image_batch1, dim=0)}, is_training=False, num_classes=args.num_classes, num_layers = n_layers)
    #
    # # Which variables to load.
    # restore_var = tf.global_variables()
    #
    # # Predictions.
    # pred_reg = net.layers['pred_up_reg']
    # pred_logit = net.layers['pred_up_class']
    # softmax_output = tf.nn.softmax(pred_logit)
    #
    #
    # pred_class = tf.argmax(pred_logit, axis=3)
    #
    #
    # # Set up TF session and initialize variables.
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # init = tf.global_variables_initializer()
    #
    # sess.run(init)
    #
    # # Load weights.
    # loader = tf.train.Saver(var_list=restore_var)
    # load(loader, sess, args.model_weights)
    #
    # # Perform inference.

    # if args.img_path.endswith('.npz'):
    #     fileList = [args.img_path]
    # else:
    #     fileList = [x for x in sorted(glob.glob(os.path.join(args.img_path, "*.npz")))]
    #     folder = fileList[0].split('/')[-2]
    #     args.save_dir = os.path.join(args.save_dir,folder)

    metrics_merged = None
    # for file in fileList:
    filename = 'test'
    # filename = os.path.split(file)[-1]
    # filename = filename.replace('.npz', '')
    save_dir = os.path.join(args.save_dir, filename, '')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #
    # save_parameters(args, save_dir, sysargv=sys.argv, name='Params')
    #
    #
    # data_orig, label, points = load_image(file, ix, iy, size)
    # feed_dict = {data_placeholder: data_orig}
    # softmax_ = sess.run(softmax_output, feed_dict=feed_dict)[0, :, :, :]
    #
    # for i in range(softmax_.shape[2]):
    #     plots.plot_rgb(softmax_[:,:,i], save_dir+'heatmap{}'.format(i), max_luminance=1)
    #
    # preds = sess.run(pred_class,feed_dict=feed_dict)

    preds_enc = Encoder.encode(preds,0)
    labels_enc = Encoder.encode(label, 0)
    plots.plot_rgb(data[:, :, 0:3], save_dir + 'RGB')
    plots.plot_labels(labels_enc, file=save_dir + 'GT', colors=Encoder.color_map)
    plots.plot_rgb_mask(data=data[:, :, 0:3], preds=labels_enc, file=save_dir + 'RGBGT',
                  colors=Encoder.color_map)


    plot_reg(points, file=save_dir + 'GT_density')
    plots.plot_rgb_density(data=data[..., 0:3], preds=points, file=save_dir + 'RGBGT_density', pred_range = pred_range,bw=True,
                 cmap='jet')

    plots.plot_labels(preds_enc, file=save_dir + 'preds', colors=Encoder.color_map)
    plots.plot_rgb_mask(data=data[...,0:3], preds=preds_enc, file=save_dir + 'RGBpreds', colors = Encoder.color_map)

    # pred_regs = sess.run(pred_reg, feed_dict=feed_dict)


    plot_reg(pred_regs, file=save_dir + 'preds_density')
    plots.plot_rgb_density(data=data[..., 0:3], preds=pred_regs, file=save_dir + 'RGBpreds_density', pred_range = pred_range,bw=True,
                 cmap='jet')
    plot_histograms(points, pred_regs, save_dir + 'Hist_plot', predrange=pred_range)


    diff = points.squeeze() - pred_regs.squeeze()
    plot_diff(abs(diff), file=save_dir + 'GT-pred_abs')
    plot_diff(diff, file=save_dir + 'GT-pred')

    metrics_new = compute_metrics(preds, label, pred_regs, points, cld=data[..., -1],
                              id_file=filename)
    write_to_file(metrics_new, filename=save_dir + 'Metrics')

    metrics_merged = merge_metrics(metrics_new,metrics_merged)



    plot_inference(points, pred_regs.squeeze(), name=save_dir+'_inference')

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    metrics_file = os.path.join(args.save_dir,'metrics_merged')
    print(args.model)
    save_obj(metrics_merged,name=metrics_file)
    write_to_file(metrics_merged,metrics_file)
    plot_colorbar(args.save_dir,pred_range)


    print(' [*] Success!')

if __name__ == '__main__':
    main()
