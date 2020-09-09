import numpy as np
import os
import argparse
import sys
import glob
import tensorflow as tf


from utils.data_reader import DataReader
from utils.utils import save_parameters, add_letter_path
from utils.trainer import Trainer
import utils.tools_tf as tools
from utils.predict_and_recompose import predict_and_recompose_individual, predict_and_recompose_individual_MC
from data_config import untar

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the patched TEST dataset.")
parser.add_argument("model_weights", type=str,nargs='+',
                    help="Path to the file with model weights.")

# Input data args
parser.add_argument("--unlabeled_data", default=None)
parser.add_argument("--roi_lon_lat_unlab", default=None)
parser.add_argument("--roi_lon_lat_test", default=None)
parser.add_argument("--dataset", default='')
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
parser.add_argument("--border", default=4, type=int, help="Border overlap between patches. N/A for random samples")
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
parser.add_argument("--n-workers", default=4, type=int, help="Number of workers for each dataset")
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

parser.add_argument("--is-use-location",default=False, action="store_true",
                    help="use patch coordinate location for training")
parser.add_argument("--fusion-type", type=str, default='concat',
                    help="['concat', 'soft', 'hard']")

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

parser.add_argument("--compression", default='0', help='compression algorithm to save geotifs')

parser.add_argument("--is-dropout-uncertainty", default=False, action="store_true",
                    help="adding dropout to cnn filters at train and test time.")    
parser.add_argument("--mc-repetitions", default=1, type=int,
                    help="Number of forward passes to do. predicts and saves only x_sum and x2_sum")
def main(args):

    args.is_hr_label = False

    args.is_upsample_LR = False

    if '*' in args.data_dir:
        foldername = '_'.join(args.data_dir.split('*')[-2:]).replace('.SAFE','')
        data_dir = glob.glob(args.data_dir)[:10]
        if len(data_dir) < 10:
            untar(file_pattern=args.data_dir)
        data_dir = glob.glob(args.data_dir)[:10]

    else:
        foldername = (args.data_dir.split('.SAFE')[0]+'.SAFE').split('/')[-1]
        foldername = foldername.split('_')[4:6]
        foldername = '_'.join(foldername)
        data_dir = [args.data_dir]

    
    tag_ = f'{args.dataset}_{args.model}'
    if args.tag is not None and args.tag != '':
        tag_ = tag_ + '_'+ args.tag
    model_dir = os.path.join(args.save_dir,tag_, foldername, '')
    print(model_dir)
    is_ensemble = False
    
    if len(args.model_weights)== 1 and '.ckpt' in args.model_weights[0]:
        ckpt = args.model_weights[0]
    else:
        ref_folder = '/*' if args.model_weights[0].endswith('last') else 'best/*'
        if len(args.model_weights) > 1:
            ckpt = [tools.get_last_best_ckpt(x, folder=ref_folder) for x in args.model_weights]
            print(f'will load {len(ckpt)} models at every forward pass')
            assert len(ckpt) == args.mc_repetitions
            is_ensemble = True
        else:
            ckpt = tools.get_last_best_ckpt(args.model_weights[0], folder=ref_folder)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    elif not args.is_overwrite_pred:
        safe_ = data_dir[0].split('/')[-1]
        file_reg = f'{model_dir}/{safe_}-test-{args.mc_repetitions}_preds_reg_{args.compression}.tif'
        if os.path.isfile(file_reg):
            print(f'{file_reg} already exists')
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

    trainer = Trainer(args, inference_only=True)
    trainer.model.inputnorm = tools.InputNorm(n_channels=13 if args.is_use_location else 11)
    if isinstance(ckpt,list):
        trainer.model.load_weights(ckpt[0])
    else:
        trainer.model.load_weights(ckpt)
    is_hr_pred = False
    if args.mc_repetitions > 1:
        assert args.is_dropout_uncertainty or is_ensemble
        type_ = f'test-{args.mc_repetitions}'
    else:
        type_ = 'test'
    tf.random.set_seed(args.numpy_seed)
    for test_ in test_dsets:
        print('processing',test_)
        args.test = [test_]
        try:
            reader = DataReader(args, datatype='test')
        except AssertionError:
            reader = None

        if reader is not None:
            # input_fn_test_comp = reader.get_input_test(is_restart=True,as_list=True)
            input_fn_test_comp = None
            if args.mc_repetitions > 1:
                predict_and_recompose_individual_MC(trainer, reader,
                            input_fn= input_fn_test_comp, patch_generator= reader.single_gen_test,
                            is_hr_pred=is_hr_pred, batch_size= args.batch_size_eval,type_=type_,
                            is_reg=(args.lambda_reg > 0.), is_sem=False,
                            chkpt_path=ckpt,return_array=False, mc_repetitions=args.mc_repetitions, is_ensemble=is_ensemble, compression=args.compression)
            else:
                predict_and_recompose_individual(trainer, reader,
                                input_fn= input_fn_test_comp, patch_generator= reader.single_gen_test,
                                is_hr_pred=is_hr_pred, batch_size= args.batch_size_eval,type_=type_,
                                is_reg=(args.lambda_reg > 0.), is_sem=False,
                                chkpt_path=ckpt,return_array=False, compression=args.compression)

if __name__ == '__main__':
    
    args = parser.parse_args()
    main(args)

print('Done!')
