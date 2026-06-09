import torch.nn as nn

def get_losses():
    seg_loss = nn.CrossEntropyLoss()
    cls_loss = nn.CrossEntropyLoss()
    return seg_loss, cls_loss
