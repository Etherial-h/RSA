import os.path as osp
import time
import random
import numpy as np
import argparse
import pickle as pkl
import gzip
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision.models as models

from models import create
from optimizer.SGD import SGDTrainer
from optimizer.SAGA import SAGATrainer
from optimizer.RSA import RSATrainer
from utils.data import IterLoader
from utils.aggregate import *
from utils.attacks import *
from utils.metric import getAccuracy


attacks = {"white": white,
           "maxValue": maxValue,
           "zeroGradient": zeroGradient}


def plot_result(epochs, train, val, label):
    plt.plot(range(epochs + 1), train, lw=2, label='training-'+label)
    plt.plot(range(epochs + 1), val, lw=2, label='validation-'+label)
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.savefig(label+".svg", format="svg")
    plt.cla()


def main():
    args = parser.parse_args()
    random.seed(0)
    main_worker(args)


def main_worker(args):
    assert args.byzantine_size == 0 or args.attack is not None
    assert args.honest_size != 0

    nodeSize = args.honest_size + args.byzantine_size

    # 模型
    #model = create(args.arch)
    model = [create(args.arch)] * (1+args.honest_size)

    # 加载数据集
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
        # Normalize a tensor image with mean 0.1307 and standard deviation 0.3081
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data/',
                                               train=True,
                                               transform=train_transform,
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/',
                                              train=False,
                                              transform=test_transform)

    # 数据分片
    pieces = [(i * len(train_dataset)) // args.honest_size for i in range(args.honest_size + 1)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    train_dataset_subset = [torch.utils.data.Subset(train_dataset, range(pieces[i], pieces[i + 1])) for i in
                            range(args.honest_size)]

    # 随机取样器
    randomSampler = lambda dataset: torch.utils.data.sampler.RandomSampler(
        dataset,
        num_samples=args.iters * args.batch_size,
        replacement=True
    )
    train_random_loaders_splited = [IterLoader(torch.utils.data.DataLoader(
        dataset=subset,
        batch_size=args.batch_size,
        sampler=randomSampler(subset),
    )) for subset in train_dataset_subset]

    #trainer = SGDTrainer(model, args.honest_size, gm, args.byzantine_size, attacks[args.attack])
    trainer = RSATrainer(model, 0.07, args.lr, args.honest_size, gm, args.byzantine_size, attacks[args.attack])
    # trainer = SAGATrainer(model, args.honest_size, gm, train_dataset, pieces, args.byzantine_size, attacks[args.attack])
    # trainer.init(train_loader)

    trainAccuracy, trainLoss = getAccuracy(model[0], train_loader, nn.CrossEntropyLoss())
    valAccuracy, valLoss= getAccuracy(model[0], test_loader, nn.CrossEntropyLoss())

    trainLossPath = [trainLoss]
    trainAccPath = [trainAccuracy]
    valLossPath = [valLoss]
    valAccPath = [valAccuracy]
    variencePath = []

    for epoch in range(args.epochs):
        for i in range(len(train_random_loaders_splited)):
            train_random_loaders_splited[i].new_epoch()

        #var = trainer.train(epoch, train_random_loaders_splited, args.lr, args.weight_decay, args.iters)
        #var = trainer.train(epoch, args.lr, args.weight_decay, args.iters)
        trainer.train(epoch, train_random_loaders_splited, args.weight_decay, args.iters)

        trainAccuracy, trainLoss = getAccuracy(model[0], train_loader, nn.CrossEntropyLoss())
        valAccuracy, valLoss = getAccuracy(model[0], test_loader, nn.CrossEntropyLoss())

        trainLossPath.append(trainLoss)
        trainAccPath.append(trainAccuracy)
        valLossPath.append(valLoss)
        valAccPath.append(valAccuracy)
        #variencePath.append(var)

        print("[{}/{}] train: loss={:.4f}, acc={:.2f}\t"
              "val: loss={:.4f}, acc={:.2f}".format(epoch, args.epochs, trainLoss, trainAccuracy, valLoss, valAccuracy))


    plot_result(args.epochs, trainLossPath, valLossPath, "loss")
    plot_result(args.epochs, trainAccPath, valAccPath, "acc")

    plt.plot(range(1, args.epochs+1), variencePath, lw=2, label='varience')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('var')
    plt.savefig("var.svg", format="svg")
    plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a classifier on Mnist")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=0)
    # model
    parser.add_argument('-a', '--arch', type=str, default='MLP')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    # train config
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--attack', type=str, default="white")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--log-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # node config
    parser.add_argument('--honest-size', type=int, default=16)
    parser.add_argument('--byzantine-size', type=int, default=4)

    main()
