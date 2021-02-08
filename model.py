import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def conv1d_bn(input_channels, output_channels):
    conv1d_layer = [nn.Conv1d(input_channels, output_channels, kernel_size=1),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(inplace=True)]
    return nn.Sequential(*conv1d_layer)

def fc_bn(input_channels, output_channels):
    fc_layer = [nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True)]
    return nn.Sequential(*fc_layer)

class TNet(nn.Module):
    
    def __init__(self, 
                 channel = 3):
        super(TNet, self).__init__()
        self.channel = channel

        self.conv1_bn = conv1d_bn(self.channel, 64)
        self.conv2_bn = conv1d_bn(64, 128)
        self.conv3_bn = conv1d_bn(128, 1024)

        self.max_pool = nn.MaxPool1d(1024)

        self.fc1_bn = fc_bn(1024, 512)
        self.fc2_bn = fc_bn(512, 256)
        self.fc3 = nn.Linear(256, self.channel*self.channel)


    def forward(self, x):
        self.batch_size = x.size(0)
        output = self.conv1_bn(x)
        output = self.conv2_bn(output)
        output = self.conv3_bn(output)

        output = self.max_pool(output)
        output = nn.Flatten()(output)

        output = self.fc1_bn(output)
        output = self.fc2_bn(output)
        output = self.fc3(output)

        iden = torch.eye(self.channel, requires_grad=True). \
                    repeat(self.batch_size, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iden = iden.to(device)
        output += iden

        return output.view(self.batch_size, self.channel, self.channel) 






