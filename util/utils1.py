import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import pdb
import numpy as np
import random

def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8889'
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank%num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    return rank, num_gpus


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class PosesCollection(object):
    def __init__(self):
        self.poses_pred = None
        self.poses_gt = None
        self.masks = None
        self.action_names = []


    def update(self, poses_pred, poses_gt, masks, action_names):
        if self.poses_pred is None:
            self.poses_pred = poses_pred
            self.poses_gt = poses_gt
            self.masks = masks
        else:
            self.poses_pred = np.concatenate((self.poses_pred , poses_pred), axis=1) # K,N,C,T
            self.poses_gt = np.concatenate((self.poses_gt, poses_gt), axis=0)  # N,C,T
            self.masks = np.concatenate((self.masks, masks), axis=0)  # N,C,T
        self.action_names += action_names


    def save_poses(self, save_name, save_dir):
        self.poses_pred = np.transpose(self.poses_pred,(0,1,3,2)) # K, N, T, C
        self.poses_gt = np.transpose(self.poses_gt, (0, 2, 1))  # N, T, C
        self.masks = np.transpose(self.masks, (0, 2, 1))  # N, T, C
        dir = os.path.join(save_dir,'%s.npy'%save_name)
        poses_dict = {'poses_pred': self.poses_pred, 'poses_gt':self.poses_gt,'masks':self.masks,'action_names':self.action_names}
        np.save(dir, poses_dict)
        print('save file %s' %dir)



