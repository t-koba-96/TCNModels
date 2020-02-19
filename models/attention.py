import torch
import torch.nn as nn
import torch.nn.functional as F




# attention layer (sigmoid)
class attention_sigmoid(nn.Module):
    def __init__(self,attention_dim):
        super(attention_sigmoid, self).__init__()
        
        self.conv1 = nn.Conv2d(attention_dim,attention_dim,3,1,1)
        self.attention_layer = nn.Sequential(
                     nn.Conv2d(attention_dim,attention_dim,3,1,1),
                     nn.InstanceNorm2d(attention_dim),
                     nn.Sigmoid()
                     )
        self.pooling =  nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
               
        y = self.attention_layer(x)

        x = F.relu((self.conv1(x))*y)

        x = self.pooling(x).view(x.size(0),-1)

        return x

