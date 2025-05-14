import os

import torch
import numpy as np
from torch.backends import cudnn

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_file_to_four(path1, path2):
    users = open(path1, 'r').readlines()
    neighs = open(path2, 'r').readlines()
    usr_item = []
    neigh_dict = {}
    items_id = []

    for user in users:
        usr = user.split('\n')[0].split(',')[1:]
        usr_item.append(usr)
        for item_id in usr:
            if item_id not in items_id:
                items_id.append(item_id)

    for neigh in neighs:
        nei = neigh.split('\n')[0].split(',')
        if nei[0] not in neigh_dict:
            neigh_dict[nei[0]] = []
        neigh_dict[nei[0]].append(nei[2:])
        for item_id in nei[2:]:
            if item_id not in items_id:
                items_id.append(item_id)

    neigh_items = []

    for neigh in neigh_dict:
        neigh_items.append(neigh_dict[neigh])

    return usr_item, items_id, neigh_items


def get_item_id(user_item, items_id, neigh_item):
    item_dict = {}  # {itemid1:item1_idx, itemid2:item2_idx, ...} (to handle those discontinuous items_id)

    for item_id, item in enumerate(items_id):
        item_dict[item] = item_id

    for user in user_item:
        for usr, item in enumerate(user):
            user[usr] = item_dict[item]

    for neighs in neigh_item:
        for neigh in neighs:
            for nei, item in enumerate(neigh):
                neigh[nei] = item_dict[item]

    item_num = len(items_id)
    return user_item, neigh_item, item_num


def get_dataset(user_item, neigh_item, item_num):
    train_data = []
    train_target_data = []
    train_neigh_data = []

    valid_data = []
    valid_target_data = []
    valid_neigh_data = []

    test_data = []
    test_target_data = []
    test_neigh_data = []

    length = len(user_item)
    position1 = length * 0.8
    position2 = length * 0.9

    for neighs in neigh_item:
        for idx, neigh in enumerate(neighs):
            neighs[idx] = neigh[:-1]

    for idx, (user, neighs) in enumerate(zip(user_item, neigh_item)):
        if idx < position1:
            train_data.append(user[:-1])
            train_target_data.append(user[-1])
            train_neigh_data.append(neighs)
        elif idx < position2:
            valid_data.append(user[:-1])
            valid_target_data.append(user[-1])
            valid_neigh_data.append(neighs)
        else:
            test_data.append(user[:-1])
            test_target_data.append(user[-1])
            test_neigh_data.append(neighs)

    return (train_data, train_target_data, train_neigh_data,
            valid_data, valid_target_data, valid_neigh_data,
            test_data, test_target_data, test_neigh_data, item_num)


def metrics(model, test_loader, topn):
    hr = 0
    ndcg = 0
    user_num = 0

    for x, y, neigh in test_loader:
        x, neigh = x.to(device), neigh.to(device)
        with torch.no_grad():
            scores = model(x, neigh)
        user_num += scores.shape[0]
        scores = scores.to(device)
        scores, topn_item_position = torch.topk(scores, topn)
        scores = scores.cpu()
        topn_item_position = topn_item_position.cpu()

        for next_item_pos, topn_item_pos in zip(y, topn_item_position):
            if next_item_pos in topn_item_pos:
                hr += 1
                topn_item_pos = topn_item_pos.detach().numpy().tolist()
                idx = topn_item_pos.index(next_item_pos)
                ndcg += np.reciprocal(np.log2(idx + 2))

    hr = hr / user_num
    ndcg = ndcg / user_num
    return hr, ndcg
