from __future__ import print_function, absolute_import
import time
import random
import numpy as np
import collections
from tqdm import tqdm

# from apex import amp

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import flatten_list, unflatten_vector
from utils.metric import getVarience


class RSATrainer(object):
    def __init__(self, model, l, gamma, honest_size, aggregate, byzantine_size=0, attack=None):
        super(RSATrainer, self).__init__()
        self.model = model
        self.l = l
        self.gamma = gamma
        self.attack = attack
        self.aggregate = aggregate
        self.criterion = nn.CrossEntropyLoss()
        self.honest_size = honest_size
        self.byzantine_size = byzantine_size
        self.node_size = self.honest_size + self.byzantine_size

    def train(self, epoch, train_loader_splited, weight_decay, train_iters=200):
        for i in range(1+self.honest_size):
            self.model[i].train()
        self.model[0].zero_grad()
        for i in tqdm(range(train_iters)):
            gamma = self.gamma #/ np.sqrt(epoch * train_iters + i + 1)
            for node in range(self.honest_size):
                images, labels = train_loader_splited[node].next()
                # predicate
                outputs = self.model[node+1](images)
                loss = self.criterion(outputs, labels)
                for (para, para0) in zip(self.model[node+1].parameters(), self.model[0].parameters()):
                    loss += self.l * torch.norm(para-para0, 1)
                # back propagate
                self.model[node+1].zero_grad()
                loss.backward()

                for pi, para in enumerate(self.model[node+1].parameters()):
                    #para.data.add_(-gamma*weight_decay, para)
                    para.data.add_(-gamma, para.grad.data)

            loss = 0.
            for para in self.model[0].parameters():
                for node in range(self.byzantine_size):
                    loss += self.l * torch.norm(para-torch.normal(0, 10000, para.size()), 1)
            loss.backward()

            for para in self.model[0].parameters():
                grad = para.grad.data
                para.data.add_(-gamma * weight_decay, para)
                para.data.add_(-gamma, grad)

