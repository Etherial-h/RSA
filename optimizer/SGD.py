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


class SGDTrainer(object):
    def __init__(self, model, honest_size, aggregate, byzantine_size=0, attack=None):
        super(SGDTrainer, self).__init__()
        self.model = model
        self.attack = attack
        self.aggregate = aggregate
        self.criterion = nn.CrossEntropyLoss()
        self.honest_size = honest_size
        self.byzantine_size = byzantine_size
        self.node_size = self.honest_size + self.byzantine_size
        self.message = [[torch.zeros_like(para, requires_grad=False) for para in model.parameters()]
                        for _ in range(self.node_size)]

    def train(self, epoch, train_loader_splited, gamma, weight_decay, train_iters=200):
        self.model.train()

        for i in tqdm(range(train_iters)):
            for node in range(self.honest_size):
                images, labels = train_loader_splited[node].next()
                # predicate
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # back propagate
                self.model.zero_grad()
                loss.backward()

                for pi, para in enumerate(self.model.parameters()):
                    self.message[node][pi] = para.grad.data.clone()
                    self.message[node][pi].data.add_(weight_decay, para)

            msg_flatten = flatten_list(self.message, self.byzantine_size)
            if(self.attack != None):
                 msg_flatten = self.attack(msg_flatten, self.byzantine_size)
            g = self.aggregate(msg_flatten)
            g = unflatten_vector(g, self.model)
            for para, grad in zip(self.model.parameters(), g):
                para.data.add_(-gamma, grad)

        return getVarience(msg_flatten[0:self.honest_size])
