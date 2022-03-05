import torch


def getVarience(msg):
    m, n = msg.size()
    mean = msg.mean(dim=0)
    msg = msg - mean.expand(m, n)
    s = torch.matmul(msg, msg.transpose(0, 1)).mean()
    return s.item()


def getAccuracy(model, data_loader, loss_func, device="cpu"):
    acc = 0
    total = 0
    loss = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss += loss_func(outputs, labels)
        _, pred = torch.max(outputs.data, dim=1)
        acc += (pred == labels).sum().item()
        total += len(labels)

    acc /= total
    loss /= total
    return acc, loss
