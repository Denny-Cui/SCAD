import os

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn

from .modules import Self_attention, Add_norm
from d2l import torch as d2l

cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SCAD(nn.Module):
    def __init__(self, input_dim, hid_dim, layer_num, item_num, seq_len, k=4):
        super(SCAD, self).__init__()
        self.item_num = item_num
        self.embedding = nn.Embedding(self.item_num, input_dim)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hid_dim, num_layers=self.layer_num,
                          batch_first=True)
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.self_att1 = Self_attention(self.input_dim, self.input_dim, self.input_dim)
        self.attention = d2l.DotProductAttention(0.5)

        self.seq_len = seq_len
        self.q1 = nn.Linear(self.seq_len, self.seq_len, bias=False)
        self.k1 = nn.Linear(input_dim, input_dim, bias=False)
        self.v1 = nn.Linear(input_dim, input_dim, bias=False)

        self.add_norm1 = Add_norm(input_dim, 0.5)
        self.add_norm2 = Add_norm(self.seq_len, 0.5)
        self.yita = nn.Parameter(torch.tensor(0.5))

        self.attention1 = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim // 2), nn.Dropout(0.5))
        self.attention2 = nn.Sequential(nn.Linear(self.hid_dim // 2, 1), nn.Dropout(0.5))
        self.mu = nn.Sequential(nn.Linear(self.input_dim, self.input_dim * self.k), nn.ReLU(),
                                nn.Linear(self.input_dim * self.k, self.input_dim))

    def tar_att1(self, x, y):
        q = self.q1(x)
        k = self.k1(y)
        v = self.v1(y)
        a = (q @ k) / np.sqrt(self.input_dim)
        a = torch.softmax(a, dim=-1)
        o = a @ v.transpose(-1, -2)
        return o

    def squashing(self, x, dim=-1):
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / (squared_norm.sqrt() + 1e-8)

    def single_dynamic_routing(self, neigh):
        b = torch.zeros(self.seq_len).to(device)  # [6]
        neigh = neigh.to(device)  # [6, 128]
        for i in range(3):
            att_w = self.tar_att1(b, neigh)  # [6]
            c = torch.softmax(b, dim=-1) * self.add_norm2(b, att_w)  # [6]
            c = torch.softmax(c, dim=-1)  # [6]
            s = c @ neigh  # [1,6] @ [6, 128] = [128]
            a = self.squashing(s)  # [128]
            b = b + a @ neigh.transpose(-1, -2)  # [6]
        return a  # [128]

    def multi_dynamic_routing(self, neigh):
        b = torch.randn(self.k, self.seq_len).to(device)  # [4, 6]
        neigh = neigh.to(device)  # [6, 128]
        for i in range(3):
            att_w = self.tar_att1(b, neigh)  # [4, 6]
            c = torch.softmax(b, dim=-1) * self.add_norm2(b, att_w)  # [4, 6]
            c = torch.softmax(c, dim=-1)  # [4, 6]
            s = c @ neigh  # [4, 128]
            a = self.squashing(s)  # [4, 128]
            if i < 2:
                b = b + a @ neigh.transpose(-1, -2)  # [4, 6]
        return a  # [4, 128]

    def usr_add_attention(self, outs):
        ret = 0
        usr_hn = outs[:, -1, :]
        usr_hn = usr_hn.to(device)
        usr_hn = self.attention1(usr_hn)
        for i in range(outs.shape[1]):
            usr_hi = outs[:, i, :]
            usr_hi = usr_hi.to(device)
            usr_hi = self.attention1(usr_hi)
            sig_hi_hn = self.sigmoid(usr_hn + usr_hi)
            att = self.attention2(sig_hi_hn).cpu()
            ret += att * outs[:, i, :].cpu()
        return ret

    def forward(self, train_data, train_neigh_data):
        train_data, train_neigh_data = train_data.to(device), train_neigh_data.to(device)
        train_data = self.embedding(train_data)
        train_neigh_data = self.embedding(train_neigh_data)

        batch_size = train_data.shape[0]
        usr_hidden = torch.zeros(self.layer_num, batch_size, self.hid_dim).contiguous().to(device)
        outs, usr_hidden = self.gru(train_data, usr_hidden)

        outs = self.usr_add_attention(outs)
        gru_out = torch.zeros(outs.shape[0], outs.shape[1]).to(device)
        gru_out2 = torch.zeros(outs.shape[0], self.k, outs.shape[1]).to(device)
        cnt = 0

        for usr, neigh in zip(outs, train_neigh_data):
            hidden = torch.zeros(self.layer_num, neigh.shape[0], self.hid_dim).contiguous().to(device)
            nei_out, hidden = self.gru(neigh, hidden)
            usr = usr.to(device)
            nei_out = torch.concat((nei_out, usr.repeat(self.seq_len - 1, 1, 1)), dim=1)
            nei_out = self.usr_add_attention(nei_out)
            nei = torch.zeros(nei_out.shape[0] + 1, nei_out.shape[1])
            nei[0] = usr

            for idx, i in enumerate(nei_out):
                nei[idx + 1] = i

            nei = nei.to(device)
            after = self.self_att1(nei)
            nei = self.add_norm1(nei, after)  # [6, 128]
            gru_out[cnt] = self.single_dynamic_routing(nei)
            gru_out2[cnt] = self.multi_dynamic_routing(nei)
            cnt += 1

        item_embedding = [i for i in range(self.item_num)]
        item_embedding = torch.LongTensor(item_embedding).to(device)
        item_embedding = self.embedding(item_embedding)

        score1 = gru_out @ item_embedding.T
        score2 = gru_out2 @ item_embedding.T
        score2 = torch.max(score2, 1).values
        score = self.yita * score1 + (1 - self.yita) * score2
        return score.cpu()
