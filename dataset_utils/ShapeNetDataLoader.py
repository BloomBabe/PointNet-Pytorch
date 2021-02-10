import numpy as np 
import math
import os
import json
from torch.utils.data import Dataset


def point_normalize(point_cloud):
    assert len(point_cloud.shape) == 2

    norm_out = point_cloud - np.mean(point_cloud, axis=0)
    norm_out /= np.max(np.linalg.norm(point_cloud, axis=1))
    return norm_out
    
    
class PointCloudDataset(Dataset):

    def __init__(self,
                 split = 'trainval',
                 root_path = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                 num_points = 1024):
        super(PointCloudDataset, self).__init__()
        self.split = split
        self.root_path = root_path
        self.num_points = num_points
        self.categ_file = os.path.join(self.root_path, 'synsetoffset2category.txt')
        self.categories = {}

        with open(self.categ_file, 'r') as f:
            for line in f:
                category, subdir = line.strip().split() 
                self.categories[category] = subdir
        self.classes = dict(zip(self.categories, range(len(self.categories))))
        
        assert self.split in ['train', 'val', 'test', 'trainval']
        self.split = ['train', 'val'] if self.split == 'trainval' else [self.split]
        split_ids = set() 
        for s in self.split:
            with open(os.path.join(self.root_path, 'train_test_split', f'shuffled_{s}_file_list.json')) as f:
                split_ids = split_ids.union(set([line.split('/')[2]for line in json.load(f)]))
        
        self.datapths = []
        for item in self.categories:
            pts_dir = os.path.join(self.root_path, self.categories[item])
            fns = sorted(os.listdir(pts_dir))
            #fns = [fn for fn in fns if ((fn[0:-4] in  split_ids))]
            for fn in fns:
                token = os.path.splitext(fn)[0]
                if token in split_ids:
                    self.datapths.append((item, os.path.join(pts_dir, token + '.txt')))
        
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __preproc__(self, file):
        pass


    def __len__(self):
        return len(self.datapths)


    def __getitem__(self, idx):
        fn = self.datapths[idx]
        cat = self.datapths[idx][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        data = np.loadtxt(fn[1]).astype(np.float32)
        point_set = data[:, 0:3]
        seg = data[:, -1].astype(np.int32)
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = point_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.num_points, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

if __name__ == '__main__':
    data = PointCloudDataset()




