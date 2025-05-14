import os
from time import ctime

import torch
from torch.backends import cudnn
from torch.utils import data as Data

from utils import metrics

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(test_data, test_target_data, test_neigh_data,
         batch_size,dataset_name, topn, model_type, model_name, model, k):
    print(f'{dataset_name} testing ...', ctime())
    test_dataset = Data.TensorDataset(test_data, test_target_data, test_neigh_data)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
    print(f'dataset: {dataset_name} model_type: {model_type}, topN: {topn}, k: {k}')

    model.to(device)
    model.load_state_dict(torch.load('./save_model/' + model_name + '.pth'))
    model.eval()
    test_hr, test_ndcg = metrics(model, test_loader, topn)
    print('test_HR:', test_hr)
    print('test_NDCG:', test_ndcg)

    with open('note.txt', 'a') as f:
        f.write(model_name + ':')
        s2 = 'test: ' + 'HR: ' + str(test_hr) + ' NDCG: ' + str(test_ndcg)
        f.write(s2 + '\n')
