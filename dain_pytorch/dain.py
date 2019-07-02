# pylint: disable = arguments-differ, invalid-name
"""
Deep adaptive input normalization for PyTorch.
"""
import torch
from torch import nn


class DAINLayer(nn.Module):
    """
    Deep Adaptive Input Normalization for time series.
    https://arxiv.org/abs/1902.07892

    Input : (B, L, C), where `B` is a batch dimension, `L` is
        the lenght of the time series, `C` is the number of
        features (channels) in the time series.
    Output : (B, L, C)
    """

    def __init__(self, num_inputs: int) -> None:
        """
        Parameters
        ----------
        num_inputs: nummber of channels
        """
        super().__init__()
        self.alpha = nn.Linear(1, num_inputs, bias=False)
        self.beta = nn.Linear(1, num_inputs)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.ones_(self.alpha.weight)
        torch.nn.init.normal_(self.beta.weight)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Expected shape is (B, L, C).
        """
        s1 = inputs.mean(dim=1, keepdim=True).transpose(1, 2)
        a = _batch_diagonal(self.alpha(s1))
        s2 = (inputs - a).mean(dim=1, keepdim=True).transpose(1, 2)
        b = _batch_diagonal(self.beta(s2)).sigmoid()
        out = (inputs - a) * b
        return out


def _batch_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get main diagonal across batches.
    """
    return torch.diagonal(tensor, dim1=-2, dim2=-1).unsqueeze(1)
