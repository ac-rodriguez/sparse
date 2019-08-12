import numpy as np
import os
import argparse
import sys
import glob
import shutil
import tensorflow as tf

from tqdm import tqdm

from data_reader import DataReader
# from deeplab_resnet.data_reader import DataReader as DataReader1
from utils import save_parameters, add_letter_path
from model import Model
import plots
import patches
from data_config import get_dataset
import tools_tf as tools
from predict_and_recompose import predict_and_recompose_individual
import gdal_processing as gp
# colormax = {2: 0.93, 4: 0.155, 8: 0.04}
# HRFILE='/home/pf/pfstaff/projects/andresro/sparse/data/coco'
# LRFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
# # POINTSFILE ='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_manual.kml'
# POINTSFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_detections.kml'

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the patched TEST dataset.")
parser.add_argument("model_weights", type=str,
                    help="Path to the file with model weights.")

# Input data args
# parser.add_argument("--HR_file", default=HRFILE)
parser.add_argument("--unlabeled_data", default=None)

# parser.add_argument("--points", default=POINTSFILE)
parser.add_argument("--roi_lon_lat_unlab", default=None)
parser.add_argument("--roi_lon_lat_test", default=None)
# parser.add_argument("--roi_lon_lat_val", default='117.81,8.82,117.84,8.88')
# # parser.add_argument("--roi_lon_lat_val_lb", default='117.820,8.848,117.834,8.854')
parser.add_argument("--dataset", default='palm')
parser.add_argument("--select_bands", default="B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12",
                    help="Select the bands. Using comma-separated band names.")
parser.add_argument("--is-padding", default=False, action="store_true",
                    help="padding train data with (patch_size-1)")
# parser.add_argument("--is-hr-label", default=False, action="store_true",
#                     help="compute label on the HR resolultion")
parser.add_argument("--is-fake-hr-label", default=False, action="store_true",
                    help="compute label on the LR resolultion and to ")
parser.add_argument("--is-noS2", default=False, action="store_true",
                    help="compute LR from HR and don't use S2")
parser.add_argument("--is-degraded-hr", dest='degraded_hr',default=False, action="store_true",
                    help="add a progressive blur to HR images, (scale 1 to low res equivalence)")
parser.add_argument("--is-adversarial", default=False, action="store_true",
                    help="use adverarial GAN-like optimization instead of GRL")
parser.add_argument("--is-multi-task", default=False, action="store_true",
                    help="use multi task optimizarion for all the losses")
parser.add_argument("--is-same-volume", default=False, action="store_true",
                    help="compute same embedding volume for LR and HR models")
parser.add_argument("--not-save-arrays",dest='save_arrays', default=True, action="store_false",
                    help="save arrays of GT and input data")
parser.add_argument("--warm-start-from", default=None, help="fine tune from MODELNAME or LOWER flag checkpoint")
parser.add_argument("--low-task-evol", default=None,type=float, help="add an increasing lambda over time for low-res task")
parser.add_argument("--high-task-evol", default=None,type=float, help="add an increasing lambda over time for high-res task")
parser.add_argument("--is-empty-aerial", default=False, action="store_true",
                    help="remove aerial data for areas without label")
parser.add_argument("--is-hr-pred", default=False, action="store_true",
                    help="predict and eval in hr")
parser.add_argument("--train-patches", default=5000, type=int,
                    help="Number of random patches extracted from train area")
parser.add_argument("--patches-with-labels", default=0.5, type=float, help="Percent of patches with labels")
parser.add_argument("--val-patches", default=2000, type=int, help="Number of random patches extracted from train area")
parser.add_argument("--numpy-seed", default=None, type=int, help="Random seed for random patches extraction")

# Training args
parser.add_argument("--patch-size", default=16, type=int, help="size of the patches to be created (low-res).")
parser.add_argument("--patch-size-eval","--patch-size-test",dest="patch_size_eval", default=16, type=int, help="size of the patches to be created (low-res).")
parser.add_argument("--scale", default=2, type=int, help="Upsampling scale to train")
parser.add_argument("--batch-size", default=32, type=int, help="Batch size for training")
parser.add_argument("--batch-size-eval","--batch-size-test",dest="batch_size_eval", default=None, type=int, help="Batch size for eval")
parser.add_argument("--lambda-sr", default=1.0, type=float, help="Lambda for semi-supervised part of the loss")
parser.add_argument("--lambda-reg", default=0.5, type=float, help="Lambda for reg vs semantic task")
parser.add_argument("--lambda-weights", default=0.001, type=float, help="Lambda for L2 weights regularizer")
# parser.add_argument("--weight-decay", type=float, default=0.0005,
#                     help="Regularisation parameter for L2-loss.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
parser.add_argument("--unlabeled-after", default=0, type=np.int64, help="Feed unlabeled data after number of iterations")
parser.add_argument("--sr-after", default=None, type=np.int64, help="Start SR task after number of iterations")
parser.add_argument("--eval-every", default=1, type=int, help="Number of epochs between evaluations")
parser.add_argument("--is-slim-eval", default=False, action="store_true",
                    help="at eval do not add DA, and feat_h architectures in the graph to speed up evaluation")
parser.add_argument("--model", default="simple",
                    help="Model Architecture to be used [deep_sentinel2, ...]")
parser.add_argument("--sigma-smooth", type=int, default=None,
                    help="Sigma smooth to apply to the GT points data.")
parser.add_argument("--sq-kernel", type=int, default=2,
                    help="Smooth with a squared kernel of N 10m pixels N instead of gaussian.")
parser.add_argument("--normalize", type=str, default='normal',
                    help="type of normalization applied to the data.")
parser.add_argument("--is-restore", "--is-reload", dest="is_restore", default=False, action="store_true",
                    help="Continue training from a stored model.")
parser.add_argument("--is-multi-gpu", default=False, action="store_true",
                    help="Add mirrored strategy for multi gpu training and eval.")
parser.add_argument("--n-channels", default=12, type=int,
                    help="Number of channels to be used from the features for training")
parser.add_argument("--scale-points", default=10, type=int,
                    help="Original Scale in which the GT points was calculated")
parser.add_argument("--l2-weights-every", default=None, type=int,
                    help="How often to update the L2 norm of the weights")
parser.add_argument("--gen-loss-every", default=2, type=int,
                    help="How often to run generator loss on adversarial settings")
parser.add_argument("--is-conv",dest='is_bilinear', default=True, action="store_false",
                    help="downsampling of HR_hat is bilinear (True) or conv (False).")
parser.add_argument("--is-out-relu", default=False, action="store_true",
                    help="Adds a Relu to the output of the reg prediction")
parser.add_argument("--is-masking", default=False, action="store_true",
                    help="adding random spatial masking to labels.")
parser.add_argument("--is-lower-bound", default=False, action="store_true",
                    help="set roi traindata to roi traindata with labels")
parser.add_argument("--semi-supervised",dest='semi', default=None,
                    help="semi-supervised task")
parser.add_argument("--distill-from", default=None, type=str,
                    help="distill from a pretrained HR based model")
parser.add_argument("--domain-loss",dest='domain', default=None,
                    help="domain transfer model  HR to LR")
parser.add_argument("--optimizer", type=str, default='adam',
                    help="['adagrad', 'adam']")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for optimizer.")
# Save args

parser.add_argument("--tag", default="",
                    help="tag to add to the model directory")
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/sparse/inference',
                    help="Path to directory where models should be saved")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")
parser.add_argument("--is-overwrite-pred", default=False, action="store_true",
                    help="overwrite predictions already in folder")
# parser.add_argument("--is-predict","--is-test", dest='is_train', default=True, action="store_false",
#                     help="Predict using an already trained model")
parser.add_argument("--is-mounted", default=False, action="store_false",
                    help="directories on a mounted loc from leonhard cluster")

args = parser.parse_args()


def main(unused_args):

    if ('HR' in args.model or 'SR' in args.model or 'DA_h' in args.model or 'B_h' in args.model) and \
            not '_l' in args.model and not args.is_fake_hr_label:
        args.is_hr_label = True
    else:
        args.is_hr_label = False

    args.is_upsample_LR = False

    if '*' in args.data_dir:
        foldername = '_'.join(args.data_dir.split('*')[-2:])
        data_dir = glob.glob(args.data_dir)[:10]
    else:
        foldername = (args.data_dir.split('.SAFE')[0]+'.SAFE').split('/')[-1]
        data_dir = [args.data_dir]

    if args.tag is None:
        args.tag = ""
        # args.save_dir = args.save_dir +'_' + args.tag
    model_dir = os.path.join(args.save_dir,args.tag, foldername, '')


    if '.ckpt' in args.model_weights:
        ckpt = args.model_weights
    else:
        ckpt = tools.get_last_best_ckpt(args.model_weights, folder='best/*')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    elif os.path.exists(model_dir+'/preds_reg.tif') and not args.is_overwrite_pred:
        print('preds_reg.tif already exists')
        sys.exit(0)

    test_dsets = [{
            'lr': x,
            'hr': None,
            'gt': None,
            'roi': args.roi_lon_lat_test,
            'tilename':foldername, # only one for now
            'roi_lb': None} for x in data_dir]

    args.tr = []
    args.val = []
    args.model_dir = model_dir
    filename = 'FLAGS_pred'
    args.is_train = False
    save_parameters(args, model_dir, sys.argv, name=filename)
    params = {}

    params['model_dir'] = model_dir
    params['args'] = args
    log_steps = 500
    if args.is_multi_gpu:
        strategy = tf.contrib.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            train_distribute=strategy, eval_distribute=strategy, log_step_count_steps=log_steps)
    else:
        run_config = tf.estimator.RunConfig(log_step_count_steps=log_steps)
    best_ckpt = True

    Model_fn = Model(params)
    model = tf.estimator.Estimator(model_fn=Model_fn.model_fn,
                                   model_dir=model_dir, config=run_config)
    is_hr_pred = Model_fn.hr_emb

    tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

    for test_ in test_dsets:
        print('processing',test_)
        args.test = [test_]
        try:
            reader = DataReader(args, is_training=False)
        except AssertionError:
            reader = None

        if reader is not None:
            input_fn_test_comp = reader.get_input_test(is_restart=True,as_list=True)

            predict_and_recompose_individual(model, reader, input_fn_test_comp, reader.single_gen_test,
                            is_hr_pred, args.batch_size_eval,'test',
                            is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0),
                            chkpt_path=ckpt,return_array=False)


if __name__ == '__main__':
    tf.app.run()

print('Done!')
