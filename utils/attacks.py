import torch


def white(messages, byzantine_size):
    mu = torch.mean(messages[0:-byzantine_size], dim=0)
    messages[-byzantine_size:].copy_(mu)
    noise = torch.randn((byzantine_size, messages.size(1)), dtype=torch.float64)
    messages[-byzantine_size:].add_(30, noise)
    return messages


def maxValue(messages, byzantine_size):
    mu = torch.mean(messages[0:-byzantine_size], dim=0)
    meliciousMessage = -10 * mu
    messages[-byzantine_size:].copy_(meliciousMessage)
    return messages


def zeroGradient(messages, byzantine_size):
    s = torch.sum(messages[0:-byzantine_size], dim=0)
    messages[-byzantine_size:].copy_(-s / byzantine_size)
    return messages
