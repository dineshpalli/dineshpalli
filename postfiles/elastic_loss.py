import torch
import torch.nn as nn

class ElasticLoss(nn.Module):
    """
    ElasticLoss combines L1 and L2 losses, providing a balance between the two,
    which can be beneficial for various learning tasks. It's governed by a mix ratio alpha.
    Attributes:
        alpha (float): The balance factor between L1 and L2 loss, with 0 <= alpha <= 1.
                       alpha = 1 means pure L1 loss, alpha = 0 means pure L2 loss.
        l1_loss (nn.Module): The L1 loss component.
        l2_loss (nn.Module): The L2 loss component.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                         'mean': the sum of the output will be divided by the number of elements in the output,
                         'sum': the output will be summed.
    """
    def __init__(self, alpha=0.5, reduction='mean'):
        """
        Initializes the ElasticLoss module.
        Args:
            alpha (float): The mix ratio between L1 and L2 loss (default: 0.5).
            reduction (str): Specifies the reduction to apply to the output (default: 'mean').
        """
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction=reduction)
        self.l2_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        """
        Forward pass for calculating the elastic loss.
        Args:
            input (torch.Tensor): The input tensor.
            target (torch.Tensor): The target tensor.
        Returns:
            torch.Tensor: The calculated elastic loss.
        """
        l1 = self.l1_loss(input, target)
        l2 = self.l2_loss(input, target)
        return self.alpha * l1 + (1 - self.alpha) * l2