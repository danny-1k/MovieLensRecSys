import torch
import torch.nn as nn


"""
Loss functions for implicit Models
"""


class PointWise(nn.Module):
    def __init__(self):
        super().__init__()

        """Logistic loss
        """

    def forward(self, positive, negative):

        positive_loss = (1-torch.sigmoid(positive))
        negative_loss = torch.sigmoid(negative)

        loss = (positive_loss+negative_loss).mean()

        return loss



class BPR(nn.Module):
    def __init__(self):
        super().__init__()

        """Bayesian Personalized Ranking pairwise loss
        """

    
    def forward(self, positive, negative):

        loss = (1-torch.sigmoid(positive-negative)).mean()

        return loss



class Hinge(nn.Module):
    def __init__(self):
        super().__init__()

        """Hinge pairwise loss
        """


    def forward(self, positive, negative):

        loss = torch.clamp(negative-positive + 1, 0).mean()

        return loss



