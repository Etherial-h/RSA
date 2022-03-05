import random
import torch
from torch import nn


def ByrdSAGA(model, gamma, aggregate, weight_decay, train_dataset, test_dataset,
        honestSize=0, byzantineSize=0, attack=None,
        rounds=10, displayInterval=1000,
        device='cpu', SEED=100, fixSeed=False,
        batchSize=5,
        **kw):
    assert byzantineSize == 0 or attack != None
    assert honestSize != 0

    if fixSeed:
        random.seed(SEED)

    nodeSize = honestSize + byzantineSize

    # 数据分片
    pieces = [(i * len(train_dataset)) // honestSize for i in range(honestSize + 1)]
    dataPerNode = [pieces[i + 1] - pieces[i] for i in range(honestSize)]

    # 回复的消息
    message = [[torch.zeros_like(para, requires_grad=False) for para in model.parameters()] for _ in range(nodeSize)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    train_dataset_subset = [torch.utils.data.Subset(train_dataset, range(pieces[i], pieces[i + 1])) for i in
                            range(honestSize)]

    # 随机取样器
    randomSampler = lambda dataset: torch.utils.data.sampler.RandomSampler(
        dataset,
        num_samples=rounds * displayInterval * batchSize,
        replacement=True  # 有放回取样
    )
    train_random_loaders_splited = [torch.utils.data.DataLoader(
        dataset=subset,
        batch_size=batchSize,
        sampler=randomSampler(subset),
    ) for subset in train_dataset_subset]
