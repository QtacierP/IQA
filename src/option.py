import argparse

parser = argparse.ArgumentParser(description='The option of IQA Processing')

parser.add_argument('--GPU', type=str, default='1', help='the CUDA device you will use')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Only Support DIRMDB firstly
parser.add_argument('--dataset', type=str, default='DRIMDB', help='Dataset for training')
parser.add_argument('--prepare', action='store_true', help='Prepare Dataset')
parser.add_argument('--data_dir', type=str, default='../data', help='data directory')

parser.add_argument('--model', type=str, default='inceptionv3', help='training model')
parser.add_argument('--pre_train', type=str, default='../model/pre_train/inception_v3/inception_v3.ckpt', help='the pre trained model directory')
parser.add_argument('--save', type=str, default='../model/inception/', help='trained model to save')


parser.add_argument('--n_color', type=int, default=3, help='the channels used in training')
parser.add_argument('--cross_validation', type=int, default=0, help='the root for cross validation')
parser.add_argument('--class_num', type=int, default=2, help='the class_num for IQA')

# Training option
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--online', type=bool, default=False, help='training online')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--epoch', type=int, default=10, help='')

args = parser.parse_args()
args.data_path = args.data_dir + '/' + args.dataset

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
