import argparse

parser = argparse.ArgumentParser(description="PyTorch code to train Temporal Unit Regression Network (TURN)")

# ============================= Path Setup =================================
parser.add_argument('--dataset', type=str, default='thumos14', choices=['activitynet1.2', 'thumos14'])
parser.add_argument('--modality', type=str, default='Flow', choices=['RGB', 'Flow'])
parser.add_argument('--train_clip_path', type=str, default="./train_val_files/val_training_samples.txt")
parser.add_argument('--background_path', type=str, default="./train_val_files/background_samples.txt")
parser.add_argument('--train_featmap_dir', type=str, default="./features/val_fc6_16_overlap0.5_denseflow/")
parser.add_argument('--test_featmap_dir', type=str, default="./features/test_fc6_16_overlap0.5_denseflow/")
parser.add_argument('--test_clip_path', type=str, default="./train_val_files/test_swin.txt")
parser.add_argument('--val_video_length', type=str, default="./train_val_files/thumos14_video_length_val.txt")
parser.add_argument('--test_video_length', type=str, default="./train_val_files/thumos14_video_length_test.txt")

# ============================= Model Configs ==============================
parser.add_argument('--tr_batch_size', default=128, type=int, metavar='N',
                    help='the mini-batch size during training (default:128)')
parser.add_argument('--ts_batch_size', default=256, type=int, metavar='N',
                    help='the mini-batch size during testing')
parser.add_argument('--lambda_reg', default=2, type=int, metavar='N',
                    help='the hyper-parameter lambda for the regression loss (default: 2)')
parser.add_argument('--unit_feature_dim', default=2048, type=int, metavar='N',
                    help='the feature dimension of each unit')
parser.add_argument('--middle_layer_dim', default=1000, type=int, metavar='N',
                    help='the dimension of the middle FC layer (default: 1000)')
parser.add_argument('--dropout', default=0.5, type=int, metavar='N',
                    help='the dropout ratio')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate (default:0.05)')
parser.add_argument('--lr_steps', default=[40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--ctx_number', '--nctx', default=4, type=int)
parser.add_argument('--unit_size', default=16, type=int, metavar='N',
                    help='the size of each unit')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')

# ============================== Other Configs ==============================
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='set the epoch number to start')
parser.add_argument('--snapshot_pref', type=str, default="")

# ================================= Monitor Configs =================================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=2, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

