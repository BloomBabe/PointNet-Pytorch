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

""" PointNet for Classification """
class PointNetClass(nn.Module):

    def __init__(self, 
                 num_classes = 10,
                 channel = 3):
        super(PointNetClass, self).__init__()
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
        return pred, matrix64x64


""" PointNet for PartSegmentation """
class PointNetPartSg(nn.Module):

    def __init__(self, 
                 num_parts = 50,
                 channel = 3):
        super(PointNetPartSg, self).__init__()
        self.channel = channel
        self.num_parts = num_parts

        self.input_transform = TNet(self.channel)
        self.conv1_bn = conv1d_bn(self.channel, 64)
        self.conv2_bn = conv1d_bn(64, 128)
        self.conv3_bn = conv1d_bn(128, 128)

        self.feature_transform = TNet(128)
        self.conv4_bn = conv1d_bn(128, 512)
        self.conv5_bn = conv1d_bn(512, 2048)
        self.max_pool = nn.MaxPool1d(2048)

        self.conv6_bn = conv1d_bn(4944, 256)
        self.conv7_bn = conv1d_bn(256, 256)
        self.conv8_bn = conv1d_bn(256, 128)

        self.conv_out = nn.Conv1d(256, num_parts, kernel_size=1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    

    def forward(self, x, label): 

        matrix3x3, output = self.input_transform(x)
        output1 = self.conv1_bn(output)
        output2 = self.conv2_bn(output1)
        output3 = self.conv3_bn(output2)

        matrix128x128, output = self.feature_transform(output3)
        output4 = self.conv4_bn(output)
        output5 = self.conv5_bn(output4)
        output = self.max_pool(output5)
        output = nn.Flatten()(output)

        output = torch.cat([output, label.squeeze(1)],1)
        expand = output.view(-1, 2048+16, 1).repeat(1, 1, x.size(2))
        concat = torch.cat([expand, output1, output2, output3, output4, output5], 1)

        output = self.conv6_bn(concat)
        output = self.conv7_bn(output)
        output = self.conv8_bn(output)
        output = self.conv_out(output)

        output = output.transpose(2, 1).contiguous().view(-1, self.num_parts)
        output = self.logsoftmax(output)
        pred = output.view(x.size(0), x.size(2), self.num_parts)

        return pred, matrix128x128

""" Loss function for PointNet """
def pointnetloss(outputs, labels, transform_matrix, alpha = 0.001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id_matrix= torch.eye(transform_matrix.size(1), requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id_matrix=id_matrix.cuda()
    diff = torch.bmm(transform_matrix, transform_matrix.transpose(1, 2))-id_matrix
    return criterion(outputs, labels) + alpha * torch.mean(torch.norm(diff))
