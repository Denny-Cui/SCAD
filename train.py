import os
from time import ctime

import torch
from torch.backends import cudnn
from torch.utils import data as Data
from torch import nn, optim
from tqdm import tqdm

from utils import metrics

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(train_data, train_target_data, train_neigh_data,
          valid_data, valid_target_data, valid_neigh_data,
          batch_size, dataset_name, topn, model_type, model, k, lr):
    train_dataset = Data.TensorDataset(train_data, train_target_data, train_neigh_data)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)

    valid_dataset = Data.TensorDataset(valid_data, valid_target_data, valid_neigh_data)
    valid_loader = Data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=4)

    print(f'dataset: {dataset_name}, model type: {model_type}, topN: {topn}, k: {k}')
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fun = loss_fun.to(device)

    print('training ... ', ctime())
    best_ndcg = 0
    best_hr = 0
    epoch = 100

    model_name = dataset_name + '_' + str(model_type) + '_' + str(topn) + '_k' + str(k)
    threshold = 3
    cnt = 0
    for i in range(epoch):
        model.train()
        loss_total = 0
        train_pbar = tqdm(train_loader, position=0, ncols=100)
        for data, target, neigh in train_pbar:
            data, target, neigh = data.to(device), target.to(device), neigh.to(device)
            model = model.to(device)
            score = model(data, neigh)
            score = score.to(device)
            loss = loss_fun(score, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            train_pbar.set_description(f'epoch [{i+1}/{epoch}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        print(i+1, loss_total / len(train_loader), ctime())

        model.eval()
        hr, ndcg = metrics(model, valid_loader, topn)
        print('valid_HR:', hr)
        print('valid_NDCG:', ndcg)

        flag = False
        if hr > best_hr:
            best_hr = hr
            cnt = 0
            flag = True
        if ndcg > best_ndcg:
            torch.save(model.state_dict(), './save_model/' + model_name + '.pth')
            best_ndcg = ndcg
            cnt = 0
            flag = True
        print('best_HR:', best_hr)
        print('best_NDCG:', best_ndcg)

        with open('./process.txt', 'a') as f:
            f.write(model_name + ' ')
            s1 = 'valid: ' + 'HR: ' + str(hr) + ' NDCG: ' + str(ndcg)
            f.write(s1 + '\n')

        if not flag:
            cnt += 1
        if cnt >= threshold:
            break
    return model_name
