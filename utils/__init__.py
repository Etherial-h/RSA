from __future__ import absolute_import

import torch


def flatten_list(message, byzatine_size):
    wList = [torch.cat([p.flatten() for p in parameters]) for parameters in message[0:-byzatine_size]]
    wList.extend([torch.zeros_like(wList[0]) for _ in range(byzatine_size)])
    wList = torch.stack(wList)
    return wList

def unflatten_vector(vector, model):
    paraGroup = []
    cum = 0
    for p in model.parameters():
        newP = vector[cum:cum+p.numel()]
        paraGroup.append(newP.view_as(p))
        cum += p.numel()
    return paraGroup
