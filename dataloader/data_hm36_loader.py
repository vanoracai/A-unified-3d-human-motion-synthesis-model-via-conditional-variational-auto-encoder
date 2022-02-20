"""
fuse training and testing
"""
import torch.utils.data as data
import numpy as np
from dataloader.common.camera import *
from dataloader.common.generator import ChunkedGenerator
from dataloader.common.h36m_dataset import Human36mDataset
import random
from util import task_pose
import os
from random import choices


class HM36(data.Dataset):
    def __init__(self, opt, train=True, actions=None):#dataset,

        self.dataset = get_dataset(opt)
        root_path = getattr(opt, 'hm36_root_path')
        self.opt = opt
        self.train = train
        self.data_name = 'h36m'
        self.keypoints_name = 'gt'
        self.root_path = root_path
        self.use_2d = False
        self.input_num = opt.fixed_input_num

        self.train_list = ['S1'] if opt.subset else ['S1','S5','S6','S7','S8']
        self.test_list = ['S9'] if opt.subset else ['S9', 'S11']

        self.downsample = 1
        self.subset = 1
        self.stride = opt.stride

        self.train_out_frame_num = opt.train_out_frame_num
        self.test_out_frame_num = opt.test_out_frame_num
        self.train_start_stride = opt.train_start_stride
        self.test_start_stride = opt.test_start_stride

        self.actions = actions

        self.pad = 0
        self.action_transfer_dict = {'Sitting':'Sit', 'Phoning':'Phone','Purchases':'Purchase', 'Walking':'Walk', 'Photo':'TakePhoto',
                                     'Directions':'Direction', 'Waiting':'Wait', 'SittingDown':'SitDown', 'Greeting':'Greet', 'Smoking':'Smoke','Posing':'Pose','Eating':'Eat'}

        self.action2id = {}
        self.id2action = {}
        for i, key in enumerate(self.actions):
            if key in self.action_transfer_dict:
                action_name = self.action_transfer_dict[key]
            else:
                action_name = key

            self.action2id[action_name] = i
            self.id2action[i] = action_name
        self.action_num = len(self.id2action)
        if self.train:
            _ = self.prepare_data(self.dataset, self.train_list)
            self.cameras_train, self.poses_train_3d, self.poses_train_2d = self.fetch(self.dataset, self.train_list,subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize, self.poses_train_3d, action_labels=None, poses_2d=self.poses_train_2d, cameras=None,
                                                out_frame_num = self.train_out_frame_num, stride=opt.stride, start_stride=self.train_start_stride, start_chosed_frame=opt.start_chosed_frame)
            if opt.rank<=0:
                print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            _ = self.prepare_data(self.dataset, self.test_list)
            self.cameras_test, self.poses_test_3d, self.poses_test_2d = self.fetch(self.dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batchSize, self.poses_test_3d, action_labels=None, poses_2d=self.poses_test_2d, cameras=None,
                                                out_frame_num = self.test_out_frame_num, stride=1, start_stride=self.test_start_stride, choosed_num=opt.choosed_num)
            self.key_index = self.generator.saved_index
            if opt.rank <= 0:
                print('INFO: Testing on {} frames'.format(self.generator.num_frames()))
    def __len__(self):
        "Figure our how many sequences we have"

        return len(self.generator.pairs)

    def __getitem__(self, index):
        input = {}
        # get poses
        data = self.load_poses_action(index)
        input['rel_poses_3d'] = data['rel_poses_3d']
        t, j, _ = input['rel_poses_3d'].shape
        input['action_vec'] = data['action_vec'].repeat(t, axis=-1).astype(np.float32)
        input['action_id'] = data['action_id']
        input['action_name'] = data['action_name']


        input['rel_poses_3d'] = input['rel_poses_3d'].reshape([t,-1]).transpose(1,0) #c, t
        # get masks
        if self.train:
            input['mask'] =self.load_train_masks(input['rel_poses_3d']).astype(np.float32)# c, t
        else:
            input['mask'] = self.load_test_masks(input['rel_poses_3d']).astype(np.float32)  # c, t

        return input

    def load_poses_action(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        data = self.generator.get_batch_sample(seq_name, start_3d, end_3d, flip, reverse)
        # remove absolute root position
        action = seq_name[1]
        data['rel_poses_3d'] = data['poses_3d'].copy()
        data['rel_poses_3d'] [:,0] = 0
        # get cleaned action_name
        action_name = self.get_cleaned_action_name(action)
        data['action_id'] = self.action2id[action_name]
        if hasattr(self.opt, 'use_one_hot') and self.opt.use_one_hot:
            data['action_vec'] = np.zeros([self.action_num, 1])
            data['action_vec'][data['action_id']] = 1

        data['action_name'] = action_name
        return data

    def get_cleaned_action_name(self, action):
        if action.find(' ') != -1:
            action_name = action[:action.find(' ')]
        else:
            action_name = action
        if action_name in self.action_transfer_dict:
            action_name = self.action_transfer_dict[action_name]
        return action_name

    def load_train_masks(self, poses_3d):
        """Load different mask types for training and testing
        can set weights for different types of masks
        """
        mask_type_index = self.opt.mask_type
        weights = self.opt.mask_weights
        weights = [weight/sum(weights) for weight in weights]
        mask_type = choices(mask_type_index, weights)[0]


        # prediction, mask from a middle point to the end
        if mask_type == 1:
            mask = task_pose.prediction_mask(poses_3d)
        # completion, mask in the middle part
        if mask_type == 2:
            mask = task_pose.random_consecutive_mask(poses_3d)

        # random discrete mask
        if mask_type == 3:
            mask = task_pose.random_discrete_mask(poses_3d)

        # random discrete mask
        if mask_type == 4:
            mask = task_pose.center_consecutive_mask(poses_3d, num_masked_frames=self.opt.num_masked_frames)

        mask = np.tile(np.expand_dims(mask, 0), (self.opt.joint_num*3, 1))
        return mask


    def load_test_masks(self, poses_3d):
        mask_type_index = self.opt.test_mask_type
        weights = self.opt.test_mask_weights
        weights = [weight/sum(weights) for weight in weights]
        mask_type = choices(mask_type_index, weights)[0]

        # prediction, mask from a middle point to the end
        if mask_type == 1:
            mask = task_pose.prediction_mask(poses_3d)
        # completion, mask in the middle part
        if mask_type == 2:
            mask = task_pose.random_consecutive_mask(poses_3d)
        # sparse commpletion
        if mask_type == 3:
            mask = task_pose.sparse_consecutive_mask(poses_3d)

        if mask_type == 4:
            mask = task_pose.center_consecutive_mask(poses_3d, num_masked_frames=self.opt.num_masked_frames)


        mask = np.tile(np.expand_dims(mask, 0), (self.opt.joint_num*3, 1))
        return mask


    def prepare_data(self, dataset, folder_list):
        if self.opt.rank<=0:
            print('Preparing data...')
        for subject in folder_list:
            if self.opt.rank <= 0:
                print('load %s' % subject)
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                    if self.keypoints_name .startswith('sh'):
                        pos_3d = np.delete(pos_3d,obj=9,axis=1)# remove neck for sh 2D detection
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d
        if self.use_2d:
            if self.opt.rank <= 0:
                print('Loading 2D detections...')
            keypoints = np.load(os.path.join(self.root_path,'data_2d_' + self.data_name + '_' + self.keypoints_name + '.npz') ,allow_pickle=True)
            keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']
            if self.keypoints_name.startswith('sh'):
                self.kps_left, self.kps_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
                self.joints_left, self.joints_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
            else:
                self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
                self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
                    dataset.skeleton().joints_right())
            keypoints = keypoints['positions_2d'].item()

            for subject in folder_list:
                assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
                for action in dataset[subject].keys():
                    assert action in keypoints[
                        subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                             subject)
                    for cam_idx in range(len(keypoints[subject][action])):

                        # We check for >= instead of == because some videos in H3.6M contain extra frames
                        mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                        assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                        if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            # Shorten sequence
                            keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

                    assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

            for subject in keypoints.keys():
                for action in keypoints[subject]:
                    for cam_idx, kps in enumerate(keypoints[subject][action]):
                        # Normalize camera frame
                        cam = dataset.cameras()[subject][cam_idx]
                        # if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                        if self.keypoints_name.startswith('sh'):
                            permute_index = [6,2,1,0,3,4,5,7,8,9,13,14,15,12,11,10]
                            kps = kps[:,permute_index,:]
                        keypoints[subject][action][cam_idx] = kps
            self.keypoints = keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        """

        :param dataset:
        :param subjects:
        :param subset:
        :param parse_3d_poses:
        :return: for each pose dict it has key(subject,action,cam_index)
        """
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in dataset[subject].keys():

                if action.find(' ') != -1:
                    action_name = action[:action.find(' ')]
                else:
                    action_name = action
                if action_name in self.actions:
                    if self.use_2d:
                        poses_2d = self.keypoints[subject][action]
                        for i in range(len(poses_2d)):  # Iterate across cameras
                            out_poses_2d[(subject, action_name, str(i))] = poses_2d[i]
                    else:
                        out_poses_2d = None

                    if subject in dataset.cameras():
                        cams = dataset.cameras()[subject]
                        if self.use_2d:
                            assert len(cams) == len(poses_2d), 'Camera count mismatch'
                        for i, cam in enumerate(cams):
                            if 'intrinsic' in cam:
                                out_camera_params[(subject, action_name, str(i))] = cam['intrinsic']

                    if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                        poses_3d = dataset[subject][action]['positions_3d']
                        if self.use_2d:
                            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                        for i in range(len(poses_3d)):  # Iterate across cameras
                            out_poses_3d[(subject, action_name, str(i))] = poses_3d[i]


        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        if self.train == False and self.stride > 1:
            # Downsample as requested

            for key in out_poses_3d.keys():
                if out_poses_2d is not None:
                    out_poses_2d[key] = out_poses_2d[key][::self.stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::self.stride]

        return out_camera_params, out_poses_3d, out_poses_2d

def get_dataset(opt):
    root_path = getattr(opt, 'hm36_root_path')
    dataset_path = os.path.join(root_path, 'data_3d_' + 'h36m' + '.npz')
    dataset = Human36mDataset(dataset_path)
    return dataset












