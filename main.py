import numpy as np
import os
import argparse
import sys
import shutil
import tensorflow as tf


from utils.data_reader import DataReader

from utils.utils import save_parameters, add_letter_path
from utils.model import Model

from data_config import get_dataset
import utils.tools_tf as tools
from utils.predict_and_recompose import predict_and_recompose_individual,predict_and_recompose

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Input data args

parser.add_argument("--unlabeled_data", default=None)

parser.add_argument("--roi_lon_lat_unlab", default=None)
parser.add_argument("--dataset", default='palm')
parser.add_argument("--select_bands", default="B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12",
                    help="Select the bands. Using comma-separated band names.")
parser.add_argument("--is-padding", default=False, action="store_true",
                    help="padding train data with (patch_size-1)")
parser.add_argument("--is-fake-hr-label", default=False, action="store_true",
                    help="compute label on the LR resolultion and to ")
parser.add_argument("--is-noS2", default=False, action="store_true",
                    help="compute LR from HR and don't use S2")
parser.add_argument("--is-degraded-hr", dest='degraded_hr',default=False, action="store_true",
                    help="add a progressive blur to HR images, (scale 1 to low res equivalence)")
parser.add_argument("--not-save-arrays",dest='save_arrays', default=True, action="store_false",
                    help="save arrays of GT and input data")
parser.add_argument("--warm-start-from", default=None, help="fine tune from MODELNAME or LOWER flag checkpoint")
parser.add_argument("--low-task-evol", default=None,type=float, help="add an increasing lambda over time for low-res task")
parser.add_argument("--high-task-evol", default=None,type=float, help="add an increasing lambda over time for high-res task")
parser.add_argument("--is-empty-aerial", default=False, action="store_true",
                    help="remove aerial data for areas without label")
parser.add_argument("--train-patches", default=5000, type=int,
                    help="Number of random patches extracted from train area")
parser.add_argument("--is-total-patches-datasets",default=False, action="store_true",
                    help="distribute N train patches over datasets according to size")

parser.add_argument("--patches-with-labels", default=0.5, type=float, help="Percent of patches with labels")
parser.add_argument("--val-patches", default=2000, type=int, help="Number of random patches extracted from train area")
parser.add_argument("--numpy-seed", default=None, type=int, help="Random seed for random patches extraction")

# Training args
parser.add_argument("--patch-size", default=16, type=int, help="size of the patches to be created (low-res).")
parser.add_argument("--border", default=4, type=int, help="Border overlap between patches. N/A for random samples")
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
parser.add_argument("--eval-every", default=1, type=int, help="Number of epochs between evaluations")
parser.add_argument("--logsteps", default=500, type=int, help="Number of steps between train logs")
parser.add_argument("--is-slim-eval", default=False, action="store_true",
                    help="at eval do not add DA, and feat_h architectures in the graph to speed up evaluation")
parser.add_argument("--model", default="simple",
                    help="Model Architecture to be used [deep_sentinel2, ...]")
parser.add_argument("--combinatorial-loss", default=None, type=np.int64, help="Add combinatorial loss at n levels of sum pooling")
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
parser.add_argument("--is-out-relu", default=False, action="store_true",
                    help="Adds a Relu to the output of the reg prediction")
parser.add_argument("--is-masking", default=False, action="store_true",
                    help="adding random spatial masking to labels.")
parser.add_argument("--is-lower-bound", default=False, action="store_true",
                    help="set roi traindata to roi traindata with labels")
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
    if args.sq_kernel <= 0: args.sq_kernel = None

    if args.sq_kernel is not None:
        args.tag = '_sq{}'.format(args.sq_kernel) + args.tag

    if args.combinatorial_loss is not None:
        args.tag = f'_combl{args.combinatorial_loss}' + args.tag

    if args.patch_size_eval is None:
        args.patch_size_eval = args.patch_size
    if args.batch_size_eval is None:
        args.batch_size_eval = args.batch_size
    if args.lambda_reg == 0.0:
        args.save_dir = args.save_dir+'_sem'
    elif args.lambda_reg == 1.0:
        args.save_dir = args.save_dir + '_reg'

    lambdas = 'Lr{:.1f}_Lw{:.4f}'.format(args.lambda_reg, args.lambda_weights)
    model_dir = os.path.join(args.save_dir, f'{args.model}/PATCH{args.patch_size}_{args.patch_size_eval}_SCALE{args.scale}_{lambdas}{args.tag}')

    if args.is_overwrite and os.path.exists(model_dir) and args.is_train:
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
    log_steps = args.logsteps
    if args.is_multi_gpu:
        strategy = tf.contrib.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            train_distribute=strategy, eval_distribute=strategy, log_step_count_steps=log_steps)
    else:
        run_config = tf.estimator.RunConfig(log_step_count_steps=log_steps)
    best_ckpt = True

    if args.is_noS2 and 'B_' in args.model:
        vars_to_warm_start = ["encode.*", "countception.*"]
    else:
        vars_to_warm_start = ["countception.*"] # with a change of sensor, encoder has different channel dimensions

    if args.warm_start_from is not None:
        warm_dir = args.warm_start_from
        if best_ckpt:
            warm_dir = tools.get_last_best_ckpt(warm_dir, 'best/*')

        warm_dir = tf.estimator.WarmStartSettings(warm_dir)
    else:
        warm_dir = None
    Model_fn = Model(params)
    model = tf.estimator.Estimator(model_fn=Model_fn.model_fn,
                                   model_dir=model_dir, config=run_config, warm_start_from=warm_dir)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)  # Show training logs.

    if args.is_train:

        reader = DataReader(args, is_training=True)
        input_fn, input_fn_val = reader.get_input_fn(is_val_random=True)
        val_iters = np.sum(reader.patch_gen_val_rand.nr_patches) // args.batch_size_eval
        train_iters = np.sum(reader.patch_gen.nr_patches) // args.batch_size

        metrics_scope = 'metrics'
        if int(args.lambda_reg) == 1:
            metric_ = metrics_scope+'/mae'
            comp_fn = lambda best, new: best > new
            best = 99999.0
        else:
            metric_ = metrics_scope+'/iou'
            comp_fn = lambda best, new: best < new
            best = 0.0
        print(best, metric_)

        epoch_ = 0
        while epoch_ < args.epochs:
            print(f'[*] EPOCH: {epoch_}/{args.epochs} [0/{train_iters}]')
            model.train(input_fn, steps=train_iters*args.eval_every)
            if epoch_ == 0: # warm settings only at the first iteration
                model._warm_start_settings = None
            epoch_ += args.eval_every
            metrics = model.evaluate(input_fn_val, steps=val_iters)
            print(metrics)
            if comp_fn(best, metrics[metric_]):
                print (f'New best at epoch {epoch_}, {metric_}:{metrics[metric_]} from {best}')
                best = metrics[metric_]
                input_fn_val_comp = reader.get_input_val(is_restart=True,as_list=True)
                predict_and_recompose(model,reader,input_fn_val_comp, reader.single_gen_val,False,args.batch_size_eval,'val',
                                            prefix='best/{}'.format(epoch_), is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0), m=metrics)

            else:
                print(f'Keeping old best {metric_}:{best}')

        plt_reg = lambda x, file: plots.plot_heatmap(x, file=file, min=-1, max=2.0, cmap='viridis')

        for i_, gen_ in enumerate(reader.single_gen):
            try:
                plots.plot_rgb(gen_.d_l1, file=model_dir + f'/sample_train_LR{i_}')
                for j in range(reader.n_classes):
                    plt_reg(gen_.label_1[...,j], model_dir + f'/sample_train_reg_label{i_}_class{j}')

            except AttributeError:
                pass

        input_fn_val_comp = reader.get_input_val(is_restart=True,as_list=True)

        predict_and_recompose(model, reader, input_fn_val_comp, reader.single_gen_val,
                                    False, args.batch_size_eval,'val',
                                    is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0),
                                    chkpt_path=tools.get_last_best_ckpt(model.model_dir, 'best/*'))
        np.save('{}/train_label'.format(model_dir), reader.labels)
        np.save('{}/val_label'.format(model_dir), reader.labels_val)
        del reader.train, reader.train_h
        del reader.val, reader.val_h
        reader.prepare_test_data()

    else:
        assert os.path.isdir(args.model_dir)
        reader = DataReader(args, is_training=False)
    input_fn_test_comp = reader.get_input_test(is_restart=True,as_list=True)

    predict_and_recompose_individual(model, reader, input_fn_test_comp, reader.single_gen_test,
                                False, args.batch_size_eval,'test',
                                is_reg=(args.lambda_reg > 0.), is_sem=(args.lambda_reg < 1.0),
                                chkpt_path=tools.get_last_best_ckpt(model.model_dir, 'best/*'))

    np.save('{}/test_label'.format(model_dir), reader.labels_test)

if __name__ == '__main__':
    tf.compat.v1.app.run()

print('Done!')
