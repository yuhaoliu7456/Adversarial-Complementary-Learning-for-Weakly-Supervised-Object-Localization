import torch.nn.functional as F
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
    
    def forward(self, logits, label):
        label = label.float()  
        loss_cls = F.multilabel_soft_margin_loss(logits[0], label)
        loss_cls_ers = F.multilabel_soft_margin_loss(logits[1], label)
        loss = loss_cls + loss_cls_ers
        return loss