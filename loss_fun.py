import torch
import numpy as np
import matplotlib.pyplot as plt

class WeightedMSE():
    def __init__(self):
        dim = 448
        x = torch.abs(torch.arange(dim).unsqueeze(0)-torch.arange(dim).unsqueeze(1))
        square_weights = self.diagonal_fun(x)
        self.weigths = square_weights[*torch.triu_indices(*square_weights.shape, 2)].unsqueeze(0).unsqueeze(-1)

        print(self.weigths.shape)

    def diagonal_fun(self, x, max_weight=36):
        return 1 + max_weight * torch.sin(x/500*torch.pi)
    
    def __call__(self, y_true:torch.Tensor, y_pred:torch.Tensor):
        return ((y_true - y_pred)**2 * self.weigths).sum()/y_true.shape[0]


loss = WeightedMSE()