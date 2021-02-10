import numpy as np 
import math
import random
import os
import json
import torch
from path import Path
from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader
from dataset_utils.preprocessing import train_transforms
from dataset_utils.ModelNet10DataLoader import PointCloudData
from model import PointNet, pointnetloss


path = Path('./data/ModelNet10')
train_ds = PointCloudData(path, transform=train_transforms())
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms())

inv_classes = {i: cat for cat, i in train_ds.classes.items()}
print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', inv_classes[train_ds[0]['category']])

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

pointnet = PointNet()
pointnet.to(device)

optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
epochs = 15
for epoch in range(epochs): 
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        pointnet.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # # save the model
        # if save:
        #     torch.save(pointnet.state_dict(), "save_"+str(epoch)".pth")