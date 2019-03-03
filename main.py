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
from model import Model
import plots
import patches


HRFILE = '/home/pf/pfstaff/projects/andresro/sparse/data/coco/3000_gsd5.0.tif'

LRFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
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
parser.add_argument("--is-padding", default=False, action="store_true",
                    help="padding train data with (patch_size-1)")
parser.add_argument("--is-hr-label", default=False, action="store_true",
                    help="compute label on the HR resolultion")
parser.add_argument("--is-empty-aerial", default=False, action="store_true",
                    help="remove aerial data for areas without label")
parser.add_argument("--train-patches",default=1000,type=int, help="Number of random patches extracted from train area")
parser.add_argument("--val-patches",default=1000,type=int, help="Number of random patches extracted from train area")
parser.add_argument("--numpy-seed",default=None,type=int, help="Random seed for random patches extraction")

# parser.add_argument("--data", default="dummy",
#     help="Dataset to be used [dummy, zrh,zrh1,]")



# Training args
parser.add_argument("--patch-size", default=32, type = int, help="size of the patches to be created (low-res).")
parser.add_argument("--patch-size-eval", default=64, type = int, help="size of the patches to be created (low-res).")
parser.add_argument("--scale",default=2,type=int, help="Upsampling scale to train")
parser.add_argument("--batch-size",default=8,type=int, help="Batch size for training")
parser.add_argument("--batch-size-eval",default=None,type=int, help="Batch size for eval")
parser.add_argument("--lambda-sr",default=1.0,type=float, help="Lambda for semi-supervised part of the loss")
parser.add_argument("--lambda-reg",default=0.5,type=float, help="Lambda for reg vs semantic task")
parser.add_argument("--lambda-weights",default=1.0,type=float, help="Lambda for L2 weights regularizer")
# parser.add_argument("--weight-decay", type=float, default=0.0005,
#                     help="Regularisation parameter for L2-loss.")
parser.add_argument("--train-iters",default=1000,type=int, help="Number of iterations to train")
parser.add_argument("--eval-every",default=600,type=int, help="Number of seconds between evaluations")
parser.add_argument("--model", default="simple",
    help="Model Architecture to be used [deep_sentinel2, ...]")
parser.add_argument("--sigma-smooth", type=int, default=None,
                        help="Sigma smooth to apply to the GT points data.")
parser.add_argument("--normalize", type=str, default='normal',
                        help="type of normalization applied to the data.")
parser.add_argument("--is-restore","--is-reload",dest="is_restore", default=False, action="store_true",
                    help="Continue training from a stored model.")
parser.add_argument("--is-multi-gpu", default=False, action="store_true",
                    help="Add mirrored strategy for multi gpu training and eval.")
parser.add_argument("--n-channels", default=12, type=int,
                    help="Number of channels to be used from the features for training")
parser.add_argument("--scale-points", default=10, type=int,
                    help="Original Scale in which the GT points was calculated")
parser.add_argument("--l2-weights-every", default=None, type=int,
                    help="How often to update the L2 norm of the weights")
parser.add_argument("--is-bilinear", default=False, action="store_true",
                    help="downsampling of HR_hat is bilinear (True) or conv (False).")
parser.add_argument("--is-masking", default=False, action="store_true",
                    help="adding random spatial masking to labels.")
parser.add_argument("--is-lower-bound", default=False, action="store_true",
                    help="set roi traindata to roi traindata with labels")
parser.add_argument("--optimizer", type=str, default='adam',
                        help="['adagrad', 'adam']")
parser.add_argument("--lr", type=float, default=2.5e-4,
                    help="Learning rate for optimizer.")
# Save args

parser.add_argument("--tag", default="",
    help="tag to add to the model directory")
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/sparse/training/snapshots',
    help="Path to directory where models should be saved")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")


parser.add_argument("--is-predict",dest='is_train', default=True, action="store_false",
                    help="Predict using an already trained model")
args = parser.parse_args()



def main(unused_args):
    if args.is_lower_bound:
        print(' [!] Train ROI changed from {} to {}\n computing lower bound.'.format(args.roi_lon_lat_tr,args.roi_lon_lat_tr_lb))
        args.roi_lon_lat_tr = args.roi_lon_lat_tr_lb
        args.tag = 'LOWER'+args.tag

    if args.roi_lon_lat_tr_lb == 'all':
        args.roi_lon_lat_tr_lb = args.roi_lon_lat_tr
        args.tag = 'allGT'+args.tag
    if args.HR_file == 'None' or args.HR_file == 'none': args.HR_file = None
    if args.patch_size_eval is None: args.patch_size_eval = args.patch_size
    if args.batch_size_eval is None: args.batch_size_eval = args.batch_size

    lambdas='Lr{:.1f}_Lsr{:.1f}_Lw{:.1f}'.format(args.lambda_reg,args.lambda_reg,args.lambda_weights)
    model_dir = os.path.join(args.save_dir,'MODEL{}_PATCH{}_{}_SCALE{}_CH{}_{}{}'.format(
        args.model,args.patch_size,args.patch_size_eval,args.scale,args.n_channels,lambdas,args.tag))

    if args.is_overwrite and os.path.exists(model_dir):
        print(' [!] Removing exsiting model and starting training from iter 0...')
        shutil.rmtree(model_dir, ignore_errors=True)
    elif not args.is_restore and args.is_train:
        model_dir = add_letter_path(model_dir, timestamp=False)

    if not os.path.exists(model_dir): os.makedirs(model_dir)


    args.model_dir = model_dir
    filename = 'FLAGS' if args.is_train else 'FLAGS_pred'
    save_parameters(args,model_dir, sys.argv, name=filename)
    params = {}



    params['model_dir'] = model_dir
    params['args'] = args


    if args.is_multi_gpu:
        strategy = tf.contrib.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            train_distribute=strategy, eval_distribute=strategy)
    else:
        run_config = tf.estimator.RunConfig(save_checkpoints_secs=args.eval_every)
    Model_fn = Model(params)
    model = tf.estimator.Estimator(model_fn=Model_fn.model_fn,
                                   model_dir=model_dir, config=run_config)

    tf.logging.set_verbosity(tf.logging.INFO)# Show training logs.

    if args.is_train:

        reader = DataReader(args, is_training=True)
        input_fn, input_fn_val = reader.get_input_fn()
        val_iters = np.ceil(np.sum(reader.patch_gen_val.nr_patches) / float(args.batch_size_eval))

        # Train model and save summaries into logdir.
        # model.train(input_fn=input_fn, steps=args.train_iters)
        # scores = model.evaluate(input_fn=input_fn_val, steps=(val_iters))

        train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=args.train_iters)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_val, steps = (val_iters), throttle_secs = args.eval_every)

        tf.estimator.train_and_evaluate(model, train_spec=train_spec, eval_spec=eval_spec)

        try:
            plots.plot_rgb(reader.patch_gen.d_l1, file=model_dir + '/sample_train_LR')
            plots.plot_heatmap(reader.patch_gen.label_1, file=model_dir + '/sample_train_reg_label', min=-1, max=1)
        except AttributeError:
            pass
        try:
            plots.plot_rgb(reader.patch_gen_val.d_l1, file=model_dir + '/sample_val_LR')
            plots.plot_heatmap(reader.patch_gen_val.label_1, file=model_dir + '/sample_val_reg_label', min=-1, max=1)
        except AttributeError:
            pass

    else:

        reader = DataReader(args,is_training=False)

    reader.non_random_patches()

    input_fn, input_fn_val = reader.get_input_fn()

    def predict(input_fn, patch_gen, sufix):
        nr_patches = patch_gen.nr_patches

        batch_idxs = (nr_patches) // args.batch_size

        if is_hr_pred:
            patch = patch_gen.patch_h
            border = patch_gen.border_lab
            if 'val' in sufix:
                ref_data = reader.val_h
            else:
                ref_data = reader.train_h
        else:
            patch = patch_gen.patch_l
            border = patch_gen.border

            if 'val' in sufix:
                ref_data = reader.val
            else:
                ref_data = reader.train

        pred_r_rec = np.empty(shape=([nr_patches, patch, patch,1]))
        pred_c_rec = np.empty(shape=([nr_patches, patch, patch]))

        preds_iter = model.predict(input_fn=input_fn, yield_single_examples=False)  # ,predict_keys=['hr_hat_rgb'])

        print('Predicting {} Patches...'.format(nr_patches))
        for idx in xrange(0, batch_idxs):
            p_ = preds_iter.next()
            start, stop = idx*args.batch_size, (idx+1)*args.batch_size

            pred_r_rec[start:stop] = p_['reg']
            pred_c_rec[start:stop] = np.argmax(p_['sem'],axis=-1)

            if start % 100 == 0:
                print(start)

        ref_size = (ref_data.shape[1], ref_data.shape[0])
        print ref_size
        ## Recompose RGB
        data_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
        plots.plot_heatmap(data_recomposed,file='{}/{}_reg_pred'.format(model_dir,sufix),min=-1,max=4)
        data_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border)
        plots.plot_heatmap(data_recomposed,file='{}/{}_sem_pred'.format(model_dir,sufix),min=-1,max=1)

    is_hr_pred = (args.model == 'simpleHR' and args.is_hr_label)

    if args.is_train:
        predict(input_fn,reader.patch_gen, sufix='train')
        predict(input_fn_val,reader.patch_gen_val, sufix='val')
    else:
        predict(input_fn,reader.patch_gen, sufix='test')

    try:
        plots.plot_rgb(reader.patch_gen.d_l1, file=model_dir + '/train_LR')
        plots.plot_heatmap(reader.patch_gen.label_1, file=model_dir + '/train_reg_label', min=-1, max=4)
    except AttributeError:
        pass
    try:
        plots.plot_rgb(reader.patch_gen_val.d_l1, file=model_dir + '/val_LR')
        plots.plot_heatmap(reader.patch_gen_val.label_1, file=model_dir + '/val_reg_label', min=-1,
                           max=1)
    except AttributeError:
        pass

                # plots.plot_rgb(data_recomposed,file=model_dir+'/data_recomposed')
#         nr_patches = reader1.patch_gen.nr_patches
#
#         batch_idxs = (nr_patches) // args.batch_size
#
#
#         pred_r_rec = np.empty(shape=([nr_patches, reader.patch_gen.patch_lab, reader.patch_gen.patch_lab,1]))
#         pred_c_rec = np.empty(shape=([nr_patches, reader.patch_gen.patch_lab, reader.patch_gen.patch_lab]))
#
#         print('Predicting {} Patches...'.format(nr_patches))
#         for idx in xrange(0, batch_idxs):
#             p_ = preds_gen.next()
#             start, stop = idx*args.batch_size, (idx+1)*args.batch_size
#
#             pred_r_rec[start:stop] = p_['reg']
#             pred_c_rec[start:stop] = np.argmax(p_['sem'],axis=-1)
#
#             if start % 100 == 0:
#                 print(start)
#         border = 4
#         ref_size = (reader.train_h.shape[1], reader.train_h.shape[0]) #TODO change for lr_lab
#         print ref_size
#         ## Recompose RGB
#         data_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border*args.scale)
#         plots.plot_heatmap(data_recomposed,file=model_dir+'/pred_reg_recomposed') #,min=-1,max=1)
#         data_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border*args.scale)
#         plots.plot_heatmap(data_recomposed,file=model_dir+'/pred_sem_recomposed') #,min=-1,max=1)
#         # plots.plot_rgb(data_recomposed,file=model_dir+'/data_recomposed')





if __name__ == '__main__':
    tf.app.run()



print('Done!')
