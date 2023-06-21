import torch
import torch.nn as nn
import math

class Distrloss_layer(nn.Module):

    def __init__(self):
        super(Distrloss_layer,self).__init__()

    def forward(self, input):
        if input.dim() != 5 and input.dim() != 3:
            raise ValueError('expected 5D or 3D input (got {}D input)'
                             .format(input.dim()))                         
        
        #distrloss = (torch.min(torch.abs(0 - input), torch.abs(1 - input)) ** 2).mean() + (input.mean()- 0.5) ** 2
        #distrloss = (torch.min(torch.abs(0 - input), torch.abs(1 - input)) ** 2).mean()
        T, B, C, H, W = input.shape
        distrloss = (input.mean() - 0.5/T) ** 2  # also can be changed to distrloss = (input.mean() - 0.5) ** 2
        
        
        return distrloss

