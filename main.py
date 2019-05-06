import numpy as np
import os
import argparse
import sys
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
# colormax = {2: 0.93, 4: 0.155, 8: 0.04}
# HRFILE='/home/pf/pfstaff/projects/andresro/sparse/data/coco'
# LRFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
# # POINTSFILE ='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_manual.kml'
# POINTSFILE = '/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_detections.kml'

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input data args
# parser.add_argument("--HR_file", default=HRFILE)
# parser.add_argument("--LR_file", default=LRFILE)
# parser.add_argument("--points", default=POINTSFILE)
# parser.add_argument("--roi_lon_lat_tr", default='117.84,8.82,117.92,8.9')
# parser.add_argument("--roi_lon_lat_tr_lb", default='117.8821,8.87414,117.891,8.8654')
# parser.add_argument("--roi_lon_lat_val", default='117.81,8.82,117.84,8.88')
# # parser.add_argument("--roi_lon_lat_val_lb", default='117.820,8.848,117.834,8.854')
parser.add_argument("--dataset", default='palm')
parser.add_argument("--select_bands", default="B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12",
                    help="Select the bands. Using comma-separated band names.")
parser.add_argument("--is-padding", default=False, action="store_true",
                    help="padding train data with (patch_size-1)")
# parser.add_argument("--is-hr-label", default=False, action="store_true",
#                     help="compute label on the HR resolultion")
parser.add_argument("--is-fake-hr-label", default=False, action="store_true",
                    help="compute label on the LR resolultion and to ")
parser.add_argument("--is-noS2", default=False, action="store_true",
                    help="compute LR from HR and don't use S2")
parser.add_argument("--is-adversarial", default=False, action="store_true",
                    help="use adverarial GAN-like optimization instead of GRL")
parser.add_argument("--is-same-volume", default=False, action="store_true",
                    help="compute same embedding volume for LR and HR models")
parser.add_argument("--not-save-arrays",dest='save_arrays', default=True, action="store_false",
                    help="save arrays of GT and input data")
parser.add_argument("--warm-start-from", default=None, help="fine tune from MODELNAME or LOWER flag checkpoint")
parser.add_argument("--low-task-evol", default=None,type=float, help="add an increasing lambda over time for low-res task")
parser.add_argument("--high-task-evol", default=None,type=float, help="add an increasing lambda over time for high-res task")
parser.add_argument("--is-empty-aerial", default=False, action="store_true",
                    help="remove aerial data for areas without label")
parser.add_argument("--train-patches", default=5000, type=int,
                    help="Number of random patches extracted from train area")
parser.add_argument("--patches-with-labels", default=0.5, type=float, help="Percent of patches with labels")
parser.add_argument("--val-patches", default=2000, type=int, help="Number of random patches extracted from train area")
parser.add_argument("--numpy-seed", default=None, type=int, help="Random seed for random patches extraction")

# Training args
parser.add_argument("--patch-size", default=16, type=int, help="size of the patches to be created (low-res).")
parser.add_argument("--patch-size-eval", default=16, type=int, help="size of the patches to be created (low-res).")
parser.add_argument("--scale", default=2, type=int, help="Upsampling scale to train")
parser.add_argument("--batch-size", default=32, type=int, help="Batch size for training")
parser.add_argument("--batch-size-eval", default=None, type=int, help="Batch size for eval")
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
                    help="semi-supervised adversarial task")
parser.add_argument("--domain-loss",dest='domain', default=None,
                    help="domain transfer model  HR to LR")
parser.add_argument("--optimizer", type=str, default='adam',
                    help="['adagrad', 'adam']")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for optimizer.")
# Save args

parser.add_argument("--tag", default="",
                    help="tag to add to the model directory")
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/sparse/training/snapshots',
                    help="Path to directory where models should be saved")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")
parser.add_argument("--is-overwrite-pred", default=False, action="store_true",
                    help="overwrite predictions already in folder")
parser.add_argument("--is-predict","--is-test", dest='is_train', default=True, action="store_false",
                    help="Predict using an already trained model")
parser.add_argument("--is-mounted", default=False, action="store_false",
                    help="directories on a mounted loc from leonhard cluster")

args = parser.parse_args()


def main(unused_args):
    if args.is_train:
        d = get_dataset(args.dataset)
    else:
        d = get_dataset(args.dataset, is_mounted=args.is_mounted)

    args.__dict__.update(d)
    if 'SR' in args.model: args.tag = '_Lsr{:.1f}'.format(args.lambda_sr) + args.tag
    if args.sq_kernel <= 0: args.sq_kernel = None

    if args.sq_kernel is not None: args.tag = '_sq{}'.format(args.sq_kernel) + args.tag
    # if args.is_hr_label:
    if ('HR' in args.model or 'SR' in args.model or 'DA_h' in args.model) and \
            not '_l' in args.model and not args.is_fake_hr_label:
        args.is_hr_label = True
        args.tag = '_hrlab' + args.tag
    else:
        args.is_hr_label = False
    if args.is_fake_hr_label: args.tag = '_fakehrlab' + args.tag
    if args.is_noS2: args.tag = '_noS2' + args.tag
    if args.is_same_volume: args.tag = '_samevol' + args.tag
    assert not (args.is_fake_hr_label and args.is_hr_label)
    if args.semi is not None: args.tag = '_'+args.semi + args.tag
    if args.domain == 'None' or args.domain == 'none': args.domain = None
    if args.domain is not None: args.tag = '_'+args.domain + args.tag

    if args.is_lower_bound:
        print(' [!] Train ROI changed from {} to {}\n computing lower bound.'.format(args.roi_lon_lat_tr,
                                                                                     args.roi_lon_lat_tr_lb))
        args.roi_lon_lat_tr = args.roi_lon_lat_tr_lb
        args.tag = 'LOWER' + args.tag

    if args.roi_lon_lat_tr_lb == 'all':
        args.roi_lon_lat_tr_lb = args.roi_lon_lat_tr
        args.tag = 'allGT' + args.tag

    if args.HR_file == 'None' or args.HR_file == 'none': args.HR_file = None
    if args.patch_size_eval is None: args.patch_size_eval = args.patch_size
    if args.batch_size_eval is None: args.batch_size_eval = args.batch_size
    if args.lambda_reg == 0.0:
        args.save_dir = args.save_dir+'_sem'
    elif args.lambda_reg == 1.0:
        args.save_dir = args.save_dir + '_reg'

    lambdas = 'Lr{:.1f}_Lw{:.4f}'.format(args.lambda_reg, args.lambda_weights)
    model_dir = os.path.join(args.save_dir, '{}/PATCH{}_{}_SCALE{}_{}{}'.format(
        args.model, args.patch_size, args.patch_size_eval, args.scale, lambdas, args.tag))

    if args.is_overwrite and os.path.exists(model_dir):
        print(' [!] Removing exsiting model and starting training from iter 0...')
        shutil.rmtree(model_dir, ignore_errors=True)
    elif not args.is_restore and args.is_train:
        model_dir = add_letter_path(model_dir, timestamp=False)

    if not args.is_train:
        ckpt = tools.get_last_best_ckpt(model_dir, folder='best/*')
        args.ckpt = ckpt
        if not args.is_overwrite_pred:
            assert not os.path.isfile(os.path.join(model_dir,'test_sem_pred.png')), 'predictions exist'
            assert not os.path.isfile(os.path.join(model_dir,'test_reg_pred.png')), 'predictions exist'

    if not os.path.exists(model_dir): os.makedirs(model_dir)

    args.model_dir = model_dir
    filename = 'FLAGS' if args.is_train else 'FLAGS_pred'
    if not args.is_restore:
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
    if args.warm_start_from is not None:
        if args.warm_start_from == 'LOWER':
            assert not args.is_lower_bound, 'warm-start only works from an already trained LOWER bound'
            warm_dir = model_dir.replace(lambdas,lambdas+'LOWER')
        else:
            warm_dir = os.path.join(args.save_dir,args.warm_start_from)
            if not os.path.isdir(warm_dir):
                warm_dir = args.warm_start_from
        warm_dir = tf.estimator.WarmStartSettings(warm_dir,vars_to_warm_start=[".*encode_same.*",".*counception.*"])
    else:
        warm_dir = None
    Model_fn = Model(params)
    model = tf.estimator.Estimator(model_fn=Model_fn.model_fn,
                                   model_dir=model_dir, config=run_config, warm_start_from=warm_dir)
    is_hr_pred = Model_fn.hr_emb

    tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

    if args.is_train:

        reader = DataReader(args, is_training=True)
        input_fn, input_fn_val = reader.get_input_fn()
        val_iters = np.ceil(np.sum(reader.patch_gen_val.nr_patches) / float(args.batch_size_eval))
        train_iters = np.ceil(np.sum(reader.patch_gen.nr_patches) / float(args.batch_size))
        # print train_iters
        # assert train_iters*args.eval_every > 200.0, 'eval every should be larger than {}'.format(np.ceil(200./train_iters))
        if int(args.lambda_reg) == 1:
            metric_ = 'metrics/mae'
            comp_fn = lambda best, new: best > new
            best = 99999.0
        else:
            metric_ = 'metrics/iou'
            comp_fn = lambda best, new: best < new
            best = 0.0
        print best, metric_
        # best_exporter = tools.BestCheckpointCopier(score_metric=metric_)


        # train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=train_iters*args.epochs)#, hooks=[hook])
        # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_val, steps=(val_iters), throttle_secs=args.eval_every,
        #                                   exporters=[best_exporter])
        # model.evaluate()
        # model.export_saved_model()
        # # best_exporter.export(model,export_path=)
        # tf.estimator.train_and_evaluate(model, train_spec=train_spec, eval_spec=eval_spec)


            # if args.domain is not None:
            #     tools.get_embeddings(hook[0], Model_fn, suffix=suffix)
        epoch_ = 0
        while epoch_ < args.epochs:
            print '[*] EPOCH: {}/{} [0/{}]'.format(epoch_,args.epochs,train_iters)
            model.train(input_fn, steps=train_iters*args.eval_every)
            epoch_+=args.eval_every
            metrics = model.evaluate(input_fn_val, steps=val_iters)
            print metrics
            if comp_fn(best, metrics[metric_]):
                print ('New best at epoch {}, {}:{} from {}'.format(epoch_,metric_,metrics[metric_],best))
                best = metrics[metric_]
                input_fn_val = reader.get_input_val()
                tools.predict_and_recompose(model,reader,input_fn_val, reader.patch_gen_val,is_hr_pred,args.batch_size_eval,'val',
                                            prefix='best/{}'.format(epoch_), is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0), m=metrics)

            else:
                print ('Keeping old best {}:{}'.format(metric_,best))

        f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
        plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, min=-1, max=2.0, cmap='viridis')

        try:
            plots.plot_rgb(reader.patch_gen.d_l1, file=model_dir + '/sample_train_LR')
            plt_reg(reader.patch_gen.label_1, model_dir + '/sample_train_reg_label')
        except AttributeError:
            pass
        try:
            plots.plot_rgb(reader.patch_gen_val.d_l1, file=model_dir + '/sample_val_LR')
            plt_reg(reader.patch_gen_val.label_1, model_dir + '/sample_val_reg_label')
        except AttributeError:
            pass

    else:
        assert os.path.isdir(args.model_dir)
        reader = DataReader(args, is_training=True) # TODO to avoid reading Train and Val sets check how to restore mean_train properly

    if args.is_train:

        tools.predict_and_recompose(model, reader, reader.get_input_val(), reader.patch_gen_val, is_hr_pred, args.batch_size_eval,
                                    'val', is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0),
                                    chkpt_path=tools.get_last_best_ckpt(model.model_dir,'best/*'))
        np.save('{}/train_label'.format(model_dir), reader.labels)
        np.save('{}/val_label'.format(model_dir), reader.labels_val)

    reader.prepare_test_data()
    input_fn_test = reader.get_input_test()
    tools.predict_and_recompose(model, reader, input_fn_test, reader.patch_gen_test, is_hr_pred, args.batch_size_eval,
                                'test', is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0),
                                chkpt_path=tools.get_last_best_ckpt(model.model_dir, 'best/*'))

    np.save('{}/test_label'.format(model_dir), reader.labels_test)

if __name__ == '__main__':
    tf.app.run()

print('Done!')
