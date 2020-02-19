import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

class Attention_Visualize(nn.Module):
    def __init__(self, SETTING, test_model):
        super(Attention_Visualize, self).__init__()
        self.attention_layer = test_model.attention.attention_layer
        self.attention_size = SETTING.attention_size

    def forward(self, x):
        x = x.permute(1,0)
        x = x.view(x.size(0),-1,self.attention_size,self.attention_size)
        output = self.attention_layer(x)
        
        return output