import numpy as np 
import math
import random
import os
import sys
import json
import torch
import argparse
from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader
from dataset_utils.preprocessing import train_transforms
from dataset_utils.ModelNet10DataLoader import PointCloudData
from model import PointNet, pointnetloss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='./data/ModelNet10', help='Path of root dataset [default: ./data/ModelNet10]')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs [default: 200]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--num_points', type=int, default=1024, help='Point number [default: 4096]')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--log_dir', type=str, default='log', help='Log dir [default: log]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
args = parser.parse_args()

PATH = args.dataset_root
LOG_DIR = args.log_dir
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
NUM_POINTS = args.num_points
LR_RATE = args.learning_rate
MOMENTUM = args.momentum
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_step




train_ds = PointCloudData(PATH, transform=train_transforms())
valid_ds = PointCloudData(PATH, folder='test')

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
                for data in valid_loader:
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