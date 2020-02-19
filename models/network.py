import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

from .attention import attention_sigmoid 

class Attention_MultiStageTCN(nn.Module):
    def __init__(self, SETTING, num_classes):
        super(Attention_MultiStageTCN, self).__init__()
        self.attention = attention_sigmoid(SETTING.attention_dim)
        self.attention_size = SETTING.attention_size
        self.stage1 = SingleStageModel(SETTING.num_layers, SETTING.num_f_maps, SETTING.features_dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(SETTING.num_layers, SETTING.num_f_maps, num_classes, num_classes)) for s in range(SETTING.num_stages-1)])

    def forward(self, x, mask):
        batch_size=x.size(0)
        clip_length=x.size(2)
        x = x.permute(0,2,1)
        x = x.view(x.size(0) * x.size(1),-1,self.attention_size,self.attention_size)
        x = self.attention(x)
        x = x.view(batch_size,clip_length,-1).permute(0,2,1)
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs



class MultiStageTCN(nn.Module):
    def __init__(self, SETTING, num_classes):
        super(MultiStageTCN, self).__init__()
        self.stage1 = SingleStageModel(SETTING.num_layers, SETTING.num_f_maps, SETTING.features_dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(SETTING.num_layers, SETTING.num_f_maps, num_classes, num_classes)) for s in range(SETTING.num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs




class SingleStageTCN(nn.Module):
    def __init__(self, SETTING, num_classes):
        super(SingleStageTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(SETTING.features_dim, SETTING.num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, SETTING.num_f_maps, SETTING.num_f_maps)) for i in range(SETTING.num_layers)])
        self.conv_out = nn.Conv1d(SETTING.num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        output = out.unsqueeze(0)
        return output




class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out




class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]