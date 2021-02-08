import numpy as np 
import math
from torch.utils.data import Dataset
import os


def point_normalize(point_cloud):
    assert len(point_cloud.shape) == 2

    norm_out = point_cloud - np.mean(point_cloud, axis=0)
    norm_out /= np.max(np.linalg.norm(point_cloud, axis=1))
    return norm_out

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces
    
class PtCloudDataset(Dataset):
    def __init__(self,
                 root_path,
                 num_points=1024):
        super(PtCloudDataset, self).__init__()
        self.root_path = root_path
        self.num_points = num_points



