import torch
from torch import nn

class L2SPLoss(nn.Module):
    def __init__(self, initial_weights, l2sp_alpha=0.01):
        """
        Initialize the L2-SP loss module.
        Instead of penalizing weights for being far from zero, L2-SP penalizes them for deviating
        from a pre-defined set of weights, often referred to as a "starting point."
        This starting point is useful for fine-tuning a pre-trained model for a new task.

        Args:
            initial_weights (list of torch.Tensor): List of tensors representing
                the initial weights of the model.

            l2sp_alpha (float): Regularization coefficient.
                Controls the strength of the L2-SP regularization.
        """
        super(L2SPLoss, self).__init__()
        # Ensure the initial weights are detached from the computational graph
        # and have no gradients
        self.initial_weights = [w.detach().clone() for w in initial_weights]
        self.l2sp_alpha = l2sp_alpha

    def forward(self, model):
        """
        Calculate the L2-SP loss for the given model.
        The L2-SP loss is computed as the sum of squared differences
        between the current weights of the model and the initial weights,
        multiplied by the regularization coefficient `alpha`.

        Args:
            model (torch.nn.Module): The model whose current weights are compared
                against the initial.

        Returns:
            torch.Tensor: The computed L2-SP regularization loss.
        """
        l2_sp_loss = 0.0
        # Iterate over the model parameters and the corresponding initial weights
        for param, initial_param in zip(model.parameters(), self.initial_weights):
            l2_sp_loss += (param - initial_param).pow(2).sum()
        return self.l2sp_alpha * l2_sp_loss
