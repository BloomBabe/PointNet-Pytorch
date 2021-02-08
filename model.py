import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F


class TNet(nn.Module):
    
    def __init__(self, 
                 channel = 3):
        super(TNet, self).__init__()
        self.channel = channel

        self.conv1 = nn.Conv1d(self.channel, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.channel*self.channel)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(1024)


    def forward(self, x):
        self.batch_size = x.size(0)
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))

        output = self.max_pool(output)
        output = nn.Flatten()(output)

        output = F.relu(self.bn4(self.fc1(output)))
        output = F.relu(self.bn5(self.fc2(output)))
        output = self.fc3(output)

        iden = torch.eye(self.channel, requires_grad=True). \
                    repeat(self.batch_size, 1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iden = iden.to(device)
        output += iden

        return output.view(self.batch_size, self.channel, self.channel) 






