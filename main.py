import argparse
import os
from time import ctime

import torch
import torch.backends.cudnn as cudnn

from model.SCAD import SCAD
from test import test
from train import train
from utils import read_file_to_four, get_item_id, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--p', default='train', help='mode: test | train')
parser.add_argument('--dataset', default='yoochoose', help='target dataset: yoochoose | lfm')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--gru_layer_number', type=int, default=2, help='the size of the GRU module')
parser.add_argument('--gru_input', type=int, default=128, help='the input feature of the GRU module')
parser.add_argument('--gru_output', type=int, default=128, help='the output feature of the GRU module')
parser.add_argument('--topN', type=int, default=5, help='number of top score items selected for evaluation')
parser.add_argument('--seq_len', type=int, default=6, help='length of sequence')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--k', type=int, default=4, help='number of interests')
args = parser.parse_args()

basic_path = './'
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


if __name__ == '__main__':
    print_configuration(args)
    print(device)

    model_type = 'SCAD'
    if args.dataset == 'yoochoose' or args.dataset == 'lfm':
        dataset = args.dataset
        topn = 5
        batch_size = 64
        gru_layer_num = 2
        gru_input_feature = 128
        gru_output_feature = 128
        seq_len = 6
        lr = 0.001
        if dataset == 'yoochoose':
            k = 3
        else:
            k = 6
    else:
        dataset = args.dataset
        topn = args.topN
        batch_size = args.batch_size
        gru_layer_num = args.gru_layer_number
        gru_input_feature = args.gru_input
        gru_output_feature = args.gru_output
        seq_len = args.seq_len
        k = args.k
        lr = args.lr

    if args.p == 'train':
        flag = True
    else:
        flag = False

    torch.manual_seed(4719)
    print(f'start: {ctime()}')
    path1 = basic_path + 'datasets/' + dataset + '/user_item_8.csv'
    path2 = basic_path + 'datasets/' + dataset + '/user_item_neigh.csv'
    user_item, items_id, neigh_item = read_file_to_four(path1, path2)
    user_item, neigh_item, item_num = get_item_id(user_item, items_id, neigh_item)
    (train_data, train_target_data, train_neigh_data,
     valid_data, valid_target_data, valid_neigh_data,
     test_data, test_target_data, test_neigh_data, item_num) = get_dataset(user_item, neigh_item, item_num)

    train_data = torch.LongTensor(train_data)  # [9190, 7]
    train_target_data = torch.LongTensor(train_target_data)  # [9190]
    train_neigh_data = torch.LongTensor(train_neigh_data)  # [9190, 5, 7]
    print('train:', train_data.shape, train_target_data.shape, train_neigh_data.shape)

    valid_data = torch.LongTensor(valid_data)  # [1149, 7]
    valid_target_data = torch.LongTensor(valid_target_data)  # [1149]
    valid_neigh_data = torch.LongTensor(valid_neigh_data)  # [1149, 5, 7]
    print('valid:', valid_data.shape, valid_target_data.shape, valid_neigh_data.shape)

    test_data = torch.LongTensor(test_data)  # [1148, 7]
    test_target_data = torch.LongTensor(test_target_data)  # [1148]
    test_neigh_data = torch.LongTensor(test_neigh_data)  # [1148, 5, 7]
    print('test:', test_data.shape, test_target_data.shape, test_neigh_data.shape)

    model = SCAD(gru_input_feature, gru_output_feature, gru_layer_num, item_num, seq_len, k)
    if flag:
        model_name = train(train_data, train_target_data, train_neigh_data,
                           valid_data, valid_target_data, valid_neigh_data,
                           batch_size, dataset, topn, model_type, model, k, lr)
        print(f'{dataset} training finished')
    else:
        model_name = dataset + '_' + str(model_type) + '_' + str(topn) + '_k' + str(k)

    test(test_data, test_target_data, test_neigh_data, batch_size, dataset, topn, model_type, model_name, model, k)
    print(f'{dataset} testing finished')
