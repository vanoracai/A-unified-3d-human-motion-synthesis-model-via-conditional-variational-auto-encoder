import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random

###################################################################
# random mask generation
###################################################################
def full_mask(poses_3d, input_num=1):
    """Generates a full mask for generation task except for the first image"""
    _, T = poses_3d.shape
    mask = np.zeros(T).astype(np.float32)
    if input_num >0:
        mask[0:input_num] = 1
    return mask

def test_interpolation_mask(poses_3d, interpolation_stride):
    """Generates a full mask for generation task except for the first image"""
    _, T = poses_3d.shape
    mask = np.zeros(T).astype(np.float32)
    mask[::interpolation_stride] = 1
    return mask


def prediction_mask(poses_3d, fixed_num=None):
    """Generates a mask from one point to the end for prediction task
        split into predict prior/predict later
    """
    _, T = poses_3d.shape
    mask = np.ones(T).astype(np.float32)
    if fixed_num is not None:
        mask[fixed_num:] = 0
    else:
        prob = random.random()
        mask = np.ones(T).astype(np.float32)
        N_mask = random.randint(1, 5)


        if prob<0.5:# pred later
            limx = T - T / (N_mask + 1)
            start = random.randint(1, int(limx))
            mask[start:] = 0
        else:# pred prior
            limx = T / (N_mask + 1)
            end = random.randint(int(limx), T-2)
            mask[:end] = 0
    return mask

def random_consecutive_mask(poses_3d):
    """Generates a random consecutive hole"""
    _, T = poses_3d.shape
    mask = np.ones(T).astype(np.float32)
    N_mask = random.randint(1, 5)
    limx = T - T / (N_mask + 1)

    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        range_x = x + random.randint(int(T / (N_mask + 7)), int(T - x))

        mask[int(x):int(range_x)] = 0
    return mask

def center_consecutive_mask(poses_3d, num_masked_frames=40):
    """Generates a random consecutive hole"""
    _, T = poses_3d.shape
    mask = np.ones(T).astype(np.float32)
    center = T//2
    half_length = num_masked_frames//2
    mask[center-half_length:center+half_length] = 0
    return mask

def sparse_consecutive_mask(poses_3d):
    """Generates a sparse consecutive hole"""
    _, T = poses_3d.shape
    mask = np.zeros(T).astype(np.float32)

    # only see three frames
    mask[0] = 1
    mask[T//2]=1
    mask[-1] = 1

    return mask

def random_discrete_mask(poses_3d):
    """Generates a discrete mask"""
    _, T = poses_3d.shape
    prob = 0.2 + 0.5*random.random()
    mask = np.random.random(T)
    # set 0 for masked area, set 1 for unmasked area
    mask = 1.0*(mask >= prob)
    return mask

def spatial_mask(mask, joint_num):
    """Generates a mask with spatial occlusion"""
    channels = joint_num*3
    T = mask.shape[0]
    spatial_masked_frames = np.where(mask == 0)[0]
    spatial_masked_frames = spatial_masked_frames[np.where(np.random.random(spatial_masked_frames.shape[0])<0.2)]
    mask_spatial = (np.random.rand(joint_num, T) > 0.3)*1.0
    mask_spatial = np.tile(np.expand_dims(mask_spatial, 1),(1,3,1)).reshape([-1,T])
    mask = np.tile(np.expand_dims(mask,0),(channels, 1)) # C, T

    # set spatial occlusions
    mask[:, spatial_masked_frames] = mask_spatial[:, spatial_masked_frames]

    return mask

def scale_img(img, size, mode='nearest'):
    scaled_img = F.interpolate(img, size=size, mode=mode)#, align_corners=True)
    return scaled_img


def scale_pyramid(pose, num_scales, mode='nearest', use_2d=False):
    scaled_poses = [pose]

    s = pose.size()

    t = s[-1]

    for i in range(1, num_scales):
        ratio = 2**i
        nt = t // ratio
        if use_2d:
            size = [s[-2], nt]
        else:
            size = [nt]
        scaled_pose = scale_img(pose, size=size, mode=mode)
        scaled_poses.append(scaled_pose)

    scaled_poses.reverse()
    return scaled_poses