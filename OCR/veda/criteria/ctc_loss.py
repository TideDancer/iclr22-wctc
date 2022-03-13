import torch
import torch.nn as nn

from .registry import CRITERIA
from .wctc import wctc_loss

@CRITERIA.register_module
class CTCLoss(nn.Module):

    def __init__(self, zero_infinity=False, blank=0, reduction='mean'):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(zero_infinity=zero_infinity,
                                    blank=blank,
                                    reduction=reduction)

    def forward(self, pred, target, target_length, batch_size):
        pred = pred.log_softmax(2)
        input_lengths = torch.full(size=(batch_size,), fill_value=pred.size(1), dtype=torch.long)
        pred_ = pred.permute(1, 0, 2)
        cost = self.criterion(log_probs=pred_,
                              targets=target.to(pred.device),
                              input_lengths=input_lengths.to(pred.device),
                              target_lengths=target_length.to(pred.device))
        return cost


@CRITERIA.register_module
class WCTCLoss(nn.Module):

    def __init__(self, zero_infinity=False, blank=0, reduction='mean'):
        super(WCTCLoss, self).__init__()
        self.blank=blank
        self.reduction=reduction

    def forward(self, pred, target, target_length, batch_size):
        pred = pred.log_softmax(2)
        input_lengths = torch.full(size=(batch_size,), fill_value=pred.size(1), dtype=torch.long)
        pred_ = pred.permute(1, 0, 2)
        cost = wctc_loss(   log_probs=pred_, 
                            targets=target.to(pred.device), 
                            input_lengths=input_lengths.to(pred.device),
                            target_lengths=target_length.to(pred.device),
                            blank=self.blank,
                            reduction=self.reduction)
        return cost
 
