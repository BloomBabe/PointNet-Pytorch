import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def conv1d_bn(input_channels, output_channels):
    conv1d_layer = [nn.Conv1d(input_channels, output_channels, kernel_size=1),
                    nn.BatchNorm1d(output_channels),
                    nn.ReLU(inplace=True)]
    return nn.Sequential(*conv1d_layer)

def fc_bn(input_channels, output_channels, dropout=False, p=0.3):
    fc_layer = [nn.Linear(input_channels, output_channels)]
    if dropout:
        fc_layer.append(nn.Dropout(p=0.3))
    fc_layer.append(nn.BatchNorm1d(output_channels))
    fc_layer.append(nn.ReLU(inplace=True))
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
                     repeat(self.batch_size, 1, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iden = iden.to(device)

        output = output.view(self.batch_size, self.channel, self.channel)
        matrix = output + iden
        output = torch.bmm(torch.transpose(x, 1, 2), matrix).transpose(1, 2)

        return  matrix, output


class PointNet(nn.Module):

    def __init__(self, 
                 num_classes = 10,
                 channel = 3):
        super(PointNet, self).__init__()
        self.channel = channel
        self.num_classes = num_classes

        self.input_transform = TNet(self.channel)
        self.conv1_bn = conv1d_bn(self.channel, 64)

        self.feature_transform = TNet(64)
        self.conv2_bn = conv1d_bn(64, 128)
        self.conv3_bn = conv1d_bn(128, 1024)
        self.max_pool = nn.MaxPool1d(1024)

        self.fc1_bn = fc_bn(1024, 512)
        self.fc2_bn = fc_bn(512, 256, dropout=True)

        self.fc3 = nn.Linear(256, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    

    def forward(self, x): 
        matrix3x3, output = self.input_transform(x)
        output = self.conv1_bn(output)

        matrix64x64, output = self.feature_transform(output)
        output = self.conv2_bn(output)
        output = self.conv3_bn(output)
        output = self.max_pool(output)
        output = nn.Flatten()(output)

        output = self.fc1_bn(output)
        output = self.fc2_bn(output)
        output = self.fc3(output)

        pred = self.logsoftmax(output)
        return pred, matrix3x3, matrix64x64

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
