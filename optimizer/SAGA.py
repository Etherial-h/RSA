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


class SAGATrainer(object):
    def __init__(self, model, honest_size, aggregate, train_dataset, pieces, byzantine_size=0, attack=None):
        super(SAGATrainer, self).__init__()
        self.model = model
        self.attack = attack
        self.aggregate = aggregate
        self.criterion = nn.CrossEntropyLoss()
        self.honest_size = honest_size
        self.byzantine_size = byzantine_size
        self.node_size = self.honest_size + self.byzantine_size
        self.message = [[torch.zeros_like(para, requires_grad=False) for para in model.parameters()]
                        for _ in range(self.node_size)]
        self.train_dataset = train_dataset
        self.pieces = pieces

        self.store = []
        self.avg = []

    def init(self, dataloader):
        self.model.train()
        for image, label in dataloader:
            pred = self.model(image)
            loss = self.criterion(pred, label)
            self.model.zero_grad()
            loss.backward()
            self.store.append([para.grad.data.clone().detach() for para in self.model.parameters()])

        for i in range(self.honest_size):
            (*tmp, ) = zip(*self.store[self.pieces[i]:self.pieces[i+1]])
            self.avg.append([sum(g) / (self.pieces[i+1] - self.pieces[i]) for g in tmp])

    def train(self, epoch, gamma, weight_decay, train_iters=200):
        self.model.train()

        for i in tqdm(range(train_iters)):
            for node in range(self.honest_size):
                index = torch.randint(self.pieces[node], self.pieces[node+1], (1,))
                images, labels = self.train_dataset[index[0]]
                labels = torch.tensor([labels])
                # predicate
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # back propagate
                self.model.zero_grad()
                loss.backward()
                l = self.pieces[node+1] - self.pieces[node]
                for pi, (para, s, g) in enumerate(zip(self.model.parameters(), self.store[index], self.avg[node])):
                    grad = para.grad.data.clone()
                    grad.add_(weight_decay, para)
                    self.message[node][pi] = grad.data - s.data + g.data
                    self.store[index][pi] = grad.data
                    self.avg[node][pi].add_(1./l, grad.data - s.data)

            msg_flatten = flatten_list(self.message, self.byzantine_size)
            if(self.attack != None):
                 msg_flatten = self.attack(msg_flatten, self.byzantine_size)
            g = self.aggregate(msg_flatten)
            g = unflatten_vector(g, self.model)
            for para, grad in zip(self.model.parameters(), g):
                para.data.add_(-gamma, grad)

        return getVarience(msg_flatten[0:self.honest_size])
