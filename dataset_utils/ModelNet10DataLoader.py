import numpy as np 
import math
import os
import json
from torch.utils.data import Dataset
from dataset_utils.preprocessing import default_transforms
#from path import Path


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointCloudData(Dataset):
    
    def __init__(self, 
                 root_dir='./data/ModelNet10', 
                 folder="train", 
                 transform=default_transforms()):
        super(PointCloudData, self).__init__()
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(self.root_dir)) 
                    if os.path.isdir(os.path.join(self.root_dir, dir))]
        self.classes = {fold: i for i, fold in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = os.path.join(self.root_dir, category, folder)
            for filename in os.listdir(new_dir):
                if filename.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = os.path.join(new_dir, filename)
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file_path):
        verts, faces = read_off(file_path)
        pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}


if __name__ == '__main__':
    data = PointCloudData()





