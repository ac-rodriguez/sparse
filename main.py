import numpy as np
import os
import argparse
import sys
import shutil
import tensorflow as tf
import tqdm

from utils.data_reader import DataReader
from utils import plots
from utils.utils import save_parameters, add_letter_path
from utils.trainer import Trainer

from data_config import get_dataset
import utils.tools_tf as tools
from utils.predict_and_recompose import predict_and_recompose_individual,predict_and_recompose, predict_and_recompose_with_metrics

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
parser.add_argument("--warm-start-from", default=None, type=str, help="fine tune from MODELNAME or LOWER flag checkpoint")
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


# active learning samples
parser.add_argument("--rand-option", default=None, type=str, help="Random realization to read")
parser.add_argument("--active-samples", default=None, type=int, help="Additional samples selected")
parser.add_argument("--is-save-data-only", default=False, action="store_true",
                    help="Save data as npz only, no training")

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
parser.add_argument("--n-workers", default=4, type=int, help="Number of workers for each dataset")
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
parser.add_argument("--is-restore", "--is-reload","--is-resume", dest="is_restore", default=False, action="store_true",
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

parser.add_argument("--is-dropout-uncertainty", default=False, action="store_true",
                    help="adding dropout to cnn filters at train and test time.")               
parser.add_argument("--n-eval-dropout", default=5, type=int, help="Number of forward passes to evaluate dropout-uncertainty model")     

parser.add_argument("--is-use-location",default=False, action="store_true",
                    help="use patch coordinate location for training")
parser.add_argument("--fusion-type", type=str, default='concat',
                    help="['concat', 'soft', 'hard']")
parser.add_argument("--is-lower-bound", default=False, action="store_true",
                    help="set roi traindata to roi traindata with labels")
parser.add_argument("--optimizer", type=str, default='adam',
                    help="['adagrad', 'adam']")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for optimizer.")
parser.add_argument("--lr-step", type=int, default=10000,
                    help="Learning rate step for SGD.")
parser.add_argument("--momentum", type=float, default=0.9,
                    help="Momentum for SGD.")
parser.add_argument("--wdecay", type=float, default=0,
                    help="wdecay for all parameters.")

# Save args

parser.add_argument("--tag", default="",
                    help="tag to add to the model directory")
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/sparse/training/snapshots',
                    help="Path to directory where models should be saved")
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="Delete model_dir before starting training from iter 0. Overrides --is-restore flag")
parser.add_argument("--is-overwrite-pred", default=False, action="store_true",
                    help="overwrite predictions already in folder")
parser.add_argument("--is-val", dest='is_train', default=True, action="store_false",
                    help="Predict using an already trained model")
parser.add_argument("--is-mounted", default=False, action="store_false",
                    help="directories on a mounted loc from leonhard cluster")


def main(args):
    if args.is_train:
        if args.rand_option is not None:
            args.dataset = f'{args.dataset}_{args.active_samples}{args.rand_option}'
        d = get_dataset(args.dataset)
    else:
        if args.rand_option is not None:
            args.dataset = f'{args.dataset}_{args.active_samples}{args.rand_option}'

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

    model_name = args.model
    if args.is_dropout_uncertainty: model_name+= '_drop'
    if args.is_use_location: model_name+= '_wloc'
    
    suffix = f'PATCH{args.patch_size}_{args.patch_size_eval}_Lr{args.lambda_reg:.1f}{args.tag}'
    model_dir = os.path.join(args.save_dir, model_name,suffix)

    if args.is_overwrite and os.path.exists(model_dir) and args.is_train:
        print(' [!] Removing exsiting model and starting training from iter 0...')
        shutil.rmtree(model_dir, ignore_errors=True)
    elif not args.is_restore and args.is_train:
        model_dir = add_letter_path(model_dir, timestamp=False)

    if not args.is_train:
        ckpt = tools.get_last_best_ckpt(model_dir, folder='last/*')
        args.ckpt = ckpt
        if not args.is_overwrite_pred:
            assert not os.path.isfile(os.path.join(model_dir,'test_sem_pred.png')), 'predictions exist'
            assert not os.path.isfile(os.path.join(model_dir,'test_reg_pred.png')), 'predictions exist'

    if not os.path.exists(model_dir): os.makedirs(model_dir)

    args.model_dir = model_dir
    filename = 'FLAGS' if args.is_train else 'FLAGS_pred'
    if args.is_restore:
        assert args.warm_start_from is None, 'use only one flag'
        restore_from = tools.get_last_best_ckpt(model_dir, folder='last/*')
        restore_opt_from = restore_from.replace('model.ckpt','optimizer.pkl')
        assert os.path.isfile(restore_opt_from)
        epoch_start = int(restore_from.split('/')[-2])
        print('starting with epoch', epoch_start)
    else:
        epoch_start = 0
        save_parameters(args, model_dir, sys.argv, name=filename)

    trainer = Trainer(args)
    metrics = None
    is_compute_scalar_metrics = False
    is_predict_and_recompose = True
    assert not (is_compute_scalar_metrics and is_predict_and_recompose), 'summaries have the same names'
    
    if args.is_train:

        reader = DataReader(args, datatype='trainval')
        if args.is_save_data_only:
            sys.exit(0)
        trainer.model.inputnorm = tools.InputNorm(reader.mean_train,reader.std_train)
        if args.warm_start_from is not None:
            trainer.model.load_weights(args.warm_start_from)
            print('loaded weights from', args.warm_start_from)

        train_ds = iter(reader.input_fn(type='train'))
        
        train_iters = np.sum(reader.patch_gen.nr_patches) // args.batch_size
        trainer.is_debug = False
        if trainer.is_debug:
            train_iters = 10
        if args.lambda_reg == 1:
            metric_ = 'val_mae'
            comp_fn = lambda best, new: best > new
            best = 99999.0
        else:
            metric_ = 'val_iou'
            comp_fn = lambda best, new: best < new
            best = 0.0
        
        epoch_iter = tqdm.trange(epoch_start,args.epochs, desc=f'epoch {epoch_start} ({metric_}:{best})')
        for epoch_ in epoch_iter:
            for id_train in tqdm.trange(train_iters, desc='train',disable=False):
                sample  = next(train_ds)
                y = sample['label']
                y_hat = trainer.train_step(sample,y)
                if args.is_restore and epoch_ == epoch_start and id_train == 0: 
                    # i.e. first iteration. we needed the first train_step to have the optimizer initialized
                    trainer.model.load_weights(restore_from)
                    trainer.load_optimizer_state(file=restore_opt_from)
                    y_hat = trainer.train_step(sample,y)
                if np.mod(id_train,args.logsteps) == 0:
                    trainer.update_sum_train(y, y_hat)
                    trainer.summaries_train(step=epoch_*train_iters+id_train)
                    
            trainer.train_writer.flush()
            trainer.reset_sum_train()

            if np.mod(epoch_,args.eval_every) == 0 or epoch_ == args.epochs-1:
                if is_compute_scalar_metrics:
                    metrics = trainer.validate_dataset_patches(epoch_, reader, args)
                if is_predict_and_recompose:
                    metrics = predict_and_recompose_with_metrics(trainer,reader,args, prefix=f'last/{epoch_}',epoch_=epoch_)
                if metrics is not None:
                    epoch_iter.set_description(f'epoch {epoch_} ({metric_}:{metrics[metric_]:.2f} prev. {best:.2f})')

                trainer.val_writer.flush()
                trainer.reset_sum_val()
                trainer.model.save_weights(f'{trainer.model_dir}/last/{epoch_}/model.ckpt')
                trainer.save_optimizer_state(f'{trainer.model_dir}/last/{epoch_}/optimizer.pkl')
                
        print('Finished training, saved last model ')
        if args.save_arrays:
            plt_reg = lambda x, file: plots.plot_heatmap(x, file=file, min=-1, max=2.0, cmap='viridis')
            for i_, gen_ in enumerate(reader.single_gen):
                try:
                    plots.plot_rgb(gen_.d_l1, file=model_dir + f'/sample_train_LR{i_}')
                    for j in range(reader.n_classes):
                        plt_reg(gen_.label_1[...,j], model_dir + f'/sample_train_reg_label{i_}_class{j}')

                except AttributeError:
                    pass
        if is_predict_and_recompose:
            metrics = predict_and_recompose_with_metrics(trainer,reader,args, prefix=f'last/{epoch_}',epoch_=epoch_)

            np.save('{}/train_label'.format(model_dir), reader.labels)
            np.save('{}/val_label'.format(model_dir), reader.labels_val)
            if len(args.test) > 0: 
                del reader.train, reader.train_h
                del reader.val, reader.val_h
                reader.prepare_test_data()
            else:
                print('done, no test data was provided')

    else:
        chkpt_path = tools.get_last_best_ckpt(args.model_dir, folder='last/*')
        reader = DataReader(args, datatype='val')

        trainer.model.inputnorm = tools.InputNorm(n_channels=13 if args.is_use_location else 11)
        trainer.model.load_weights(chkpt_path)

        epoch_ = int(chkpt_path.split('/')[-2])
        if is_compute_scalar_metrics:
            metrics = trainer.validate_dataset_patches(epoch_, reader, args)
        if is_predict_and_recompose:
            metrics = predict_and_recompose_with_metrics(trainer,reader,args, prefix=f'last/{epoch_}',epoch_=epoch_)
        trainer.val_writer.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

print('Done!')
