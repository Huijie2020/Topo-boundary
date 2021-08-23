import torch
from torch.autograd import Function
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class var_loss(nn.Module):
    def __init__(self):
        super(var_loss, self).__init__()

    # def forward(self, input1, input2, input3, input4):
    #     input_mean = torch.mean(torch.stack([input1, input2, input3, input4]), 0)
    #     input_var = ((input1-input_mean)**2 + (input2-input_mean)**2 + (input3-input_mean)**2 + (input4-input_mean)**2)/3
    #     res = torch.sum(input_var)
    #     return res
    def forward(self, input1, input2):
        input_mean = torch.mean(torch.stack([input1, input2]), 0)
        input_var = ((input1-input_mean)**2 + (input2-input_mean)**2 )
        res = torch.sum(input_var)
        return res