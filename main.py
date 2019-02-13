import numpy as np
import os
import argparse
import sys
import shutil
import tensorflow as tf
from functools import partial

from data_reader import DataReader
# from deeplab_resnet.data_reader import DataReader as DataReader1
from utils import save_parameters, add_letter_path
from model import model_fn
import plots
import patches


HRFILE = '/home/pf/pfstaff/projects/andresro/sparse/data/3000_gsd5.0.tif'

LRFILE = "/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml"
# POINTSFILE ='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_manual.kml'
POINTSFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_detections.kml'

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input data args
parser.add_argument("--HR_file", default=HRFILE)
parser.add_argument("--LR_file", default=LRFILE)
parser.add_argument("--points", default=POINTSFILE)
parser.add_argument("--roi_lon_lat_tr", default='117.84,8.82,117.92,8.9')
parser.add_argument("--roi_lon_lat_tr_lb", default='117.8821,8.87414,117.891,8.8654')
parser.add_argument("--roi_lon_lat_val", default='117.81,8.82,117.84,8.88')
# parser.add_argument("--roi_lon_lat_val_lb", default='117.820,8.848,117.834,8.854')
parser.add_argument("--roi_lon_lat_val_lb", default='117.81,8.82,117.84,8.88')
parser.add_argument("--select_bands", default="B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12",
                    help="Select the bands. Using comma-separated band names.")
# parser.add_argument("--data", default="dummy",
#     help="Dataset to be used [dummy, zrh,zrh1,]")



# Training args
parser.add_argument("--patch-size", default=128, type = int, help="size of the patches to be created (low-res).")
parser.add_argument("--patch-size-eval", default=None, type = int, help="size of the patches to be created (low-res).")
parser.add_argument("--scale",default=2,type=int, help="Upsampling scale to train")
parser.add_argument("--batch-size",default=10,type=int, help="Batch size for training")
parser.add_argument("--lambda-semi",default=1,type=int, help="Lambda for semi-supervised part of the loss")
parser.add_argument("--lambda-reg",default=0.5,type=float, help="Lambda for reg vs semantic task")
parser.add_argument("--weight-decay", type=float, default=0.0005,
                    help="Regularisation parameter for L2-loss.")
parser.add_argument("--train-iters",default=1000,type=int, help="Number of iterations to train")
parser.add_argument("--eval-every",default=600,type=int, help="Number of seconds between evaluations")
parser.add_argument("--model", default="1",
    help="Model Architecture to be used [deep_sentinel2, ...]")
parser.add_argument("--sigma-smooth", type=int, default=None,
                        help="Sigma smooth to apply to the GT points data.")
parser.add_argument("--normalize", type=str, default='normal',
                        help="type of normalization applied to the data.")
parser.add_argument("--is-restore", default=False, action="store_true",
                    help="Continue training from a stored model.")
parser.add_argument("--is-multi-gpu", default=False, action="store_true",
                    help="Add mirrored strategy for multi gpu training and eval.")
parser.add_argument("--n-channels", default=12, type=int,
                    help="Number of channels to be used from the features for training")
parser.add_argument("--scale-points", default=10, type=int,
                    help="Original Scale in which the GT points was calculated")


# Save args

parser.add_argument("--tag", default="",
    help="tag to add to the model directory")
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/sparse/training/snapshots',
    help="Path to directory where models should be saved")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")


parser.add_argument("--is-predict", default=False, action="store_true",
                    help="Predict using an already trained model")
args = parser.parse_args()



def main(unused_args):

    if args.roi_lon_lat_tr_lb == 'all':
        args.roi_lon_lat_tr_lb = args.roi_lon_lat_tr
        args.tag = 'allGT'+args.tag
    if args.HR_file == 'None' or args.HR_file == 'none': args.HR_file = None
    if args.patch_size_eval is None: args.patch_size_eval = args.patch_size

    model_dir = os.path.join(args.save_dir,'model-{}_size-{}_scale-{}_nchan{}{}'.format(args.model,args.patch_size, args.scale,args.n_channels,args.tag))

    if args.is_overwrite and os.path.exists(model_dir):
        print(' [!] Removing exsiting model and starting trainign from iter 0...')
        shutil.rmtree(model_dir, ignore_errors=True)
    elif not (args.is_restore or args.is_predict):
        model_dir = add_letter_path(model_dir, timestamp=False)

    if not os.path.exists(model_dir): os.makedirs(model_dir)



    save_parameters(args,model_dir, sys.argv)
    params = {}



    params['model_dir'] = model_dir
    params['args'] = args


    if args.is_multi_gpu:
        strategy = tf.contrib.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            train_distribute=strategy, eval_distribute=strategy)
    else:
        run_config = tf.estimator.RunConfig(save_checkpoints_secs=args.eval_every)

    model = tf.estimator.Estimator(model_fn=partial(model_fn,params=params),
                                   model_dir=model_dir, config=run_config)

    tf.logging.set_verbosity(tf.logging.INFO)# Show training logs.

    if not args.is_predict:

        reader = DataReader(args, is_training=True)
        input_fn, input_fn_val = reader.get_input_fn()
        # val_iters = np.ceil(np.sum(reader.patch_gen_val.nr_patches) / float(args.batch_size))
        val_iters = 10
        # Train model and save summaries into logdir.
        # model.train(input_fn=input_fn, steps=args.train_iters)
        # scores = model.evaluate(input_fn=input_fn_val, steps=(val_iters))

        train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=args.train_iters)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_val, steps = (val_iters), throttle_secs = args.eval_every)

        tf.estimator.train_and_evaluate(model, train_spec=train_spec, eval_spec=eval_spec)
    else:
        #TODO check data_recompose output
        reader = DataReader(args,is_training=False)

        preds_gen = model.predict(input_fn=reader.input_fn_test) #,predict_keys=['hr_hat_rgb'])

        nr_patches = reader.patch_gen_test.nr_patches
        batch_idxs = (nr_patches) // args.batch_size

        rgb_rec = np.empty(shape=([nr_patches, args.patch_size*args.scale, args.patch_size*args.scale,3]))
        # pred_c_rec = np.empty(shape=([nr_patches, args.patch_size, args.patch_size]))
        # pred_r_rec = np.empty(shape=([nr_patches, args.patch_size, args.patch_size,1]))
        print('Predicting {} Patches...'.format(nr_patches))
        for idx in xrange(0, batch_idxs):
        # for idx in xrange(0,100):
            p_ = preds_gen.next()
            start, stop = idx*args.batch_size, (idx+1)*args.batch_size
            rgb_rec[start:stop] = p_['y_hat']
            # pred_c_rec[start:stop] = pc_
            # pred_r_rec[start:stop] = p_

            if start % 1000 == 0:
                print(start)
            # recompose
        border = 4
        ref_size = (args.scale*(reader.train.shape[1]-border*2), args.scale*(reader.train.shape[0]-border*2))
        ## Recompose RGB
        data_recomposed = patches.recompose_images(rgb_rec, size=ref_size, border=border)
        plots.plot_rgb(data_recomposed,file=model_dir+'/data_recomposed')

        # save

if __name__ == '__main__':
    tf.app.run()



print('Done!')
