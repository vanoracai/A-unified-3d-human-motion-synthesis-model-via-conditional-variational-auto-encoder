#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch.autograd.variable import Variable
import os
from util import forward_kinematics
from numpy import linalg as LA


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T;

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps);

    costheta = (np.trace(R) - 1) / 2;

    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R));


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use



def define_actions(action, dataset):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    if dataset == 'hm36':
        return define_actions_hm36(action)





def define_actions_hm36(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all",'distinguished (obvious actions)'
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ['Sitting', 'WalkDog', 'Phoning', 'Purchases', 'Directions', 'Waiting',
             'Greeting', 'Eating', 'Discussion', 'Walking', 'SittingDown', 'Smoking',
             'Posing',  'WalkTogether', 'Photo']
    if action == 'distinguished':
        return ['Sitting', 'Phoning', 'Walking', 'Photo',
              'Directions', 'SittingDown', 'Smoking','Eating']

    if action == "all":
        return actions

    raise (ValueError, "Unrecognized action: %d" % action)
    
def get_action2id_id2action(actions, dataset):
    if dataset == 'hm36_mvrot':
        return get_action2id_id2action_hm36_mvrot(actions)
    elif dataset == 'hm36':
        return get_action2id_id2action_hm36(actions)

def get_action2id_id2action_hm36_mvrot(actions):
    action_transfer_dict = {'sitting': 'Sit', 'phoning': 'Phone', 'purchases': 'Purchase', 'walking': 'Walk',
                                 'takingphoto': 'TakePhoto',
                                 'directions': 'Direction', 'waiting': 'Wait', 'sittingdown': 'SitDown',
                                 'greeting': 'Greet', 'smoking': 'Smoke', 'posing': 'Pose', 'eating': 'Eat',
                                 'walkingdog': 'WalkDog', 'walkingtogether': 'WalkTogether', 'discussion': 'Discussion'}
    action2id = {}
    id2action = {}
    for i, key in enumerate(actions):
        action_name = action_transfer_dict[key]
        action2id[action_name] = i
        id2action[i] = action_name
    return action2id, id2action, action_transfer_dict

def get_action2id_id2action_hm36(actions):
    action_transfer_dict = {'Sitting': 'Sit', 'Phoning': 'Phone', 'Purchases': 'Purchase', 'Walking': 'Walk',
                                 'Photo': 'TakePhoto',
                                 'Directions': 'Direction', 'Waiting': 'Wait', 'SittingDown': 'SitDown',
                                 'Greeting': 'Greet', 'Smoking': 'Smoke', 'Posing': 'Pose', 'Eating': 'Eat'}
    action2id = {}
    id2action = {}
    for i, key in enumerate(actions):
        action_name = action_transfer_dict[key]
        action2id[action_name] = i
        id2action[i] = action_name
    return action2id, id2action, action_transfer_dict






def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = Variable(torch.zeros(R.shape[0], 4)).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables_cmu()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def load_data(path_to_dataset, subjects, actions, sample_rate, seq_len, input_n=10, data_mean=None, data_std=None, action_label_dict=None, num_classes= 1, split=0):
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216

    :param path_to_dataset: path of dataset
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len: past frame length + future frame length
    :param is_norm: normalize the expmap or not
    :param data_std: standard deviation of the expmap
    :param data_mean: mean of the expmap
    :param input_n: past frame length
    :return:
    """


    sampled_seq = []
    complete_seq = []
    action_label_seq = []
    # actions_all = define_actions("all")
    # one_hot_all = np.eye(len(actions_all))
    for subj in subjects:
        print("Reading subject {0}".format(subj))
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    the_sequence = np.array(action_sequence[even_list, :])
                    num_frames = len(the_sequence)
                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]

                    ## for action label
                    # action_label = torch.zeros(1, num_classes)
                    # action_label.scatter_(1, torch.tensor(action_label_dict[action]).view(1, -1), 1).long()
                    # action_label = action_label.repeat(seq_sel.shape[0], 1)
                    action_label = torch.tensor(action_label_dict[action]).repeat(seq_sel.shape[0])

                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                        action_label_seq = action_label
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
                        action_label_seq = torch.cat((action_label_seq, action_label), 0)

            else:
                # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence1 = np.array(action_sequence[even_list, :])
                num_frames1 = len(the_sequence1)

                # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence2 = np.array(action_sequence[even_list, :])
                num_frames2 = len(the_sequence2)

                fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len, input_n=input_n)
                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]

                ## for action label
                # action_label = torch.zeros(1, num_classes)
                # action_label.scatter_(1, torch.tensor(action_label_dict[action]).view(1, -1), 1) # get one hot label
                # action_label = action_label.repeat(seq_sel1.shape[0]+seq_sel2.shape[0], 1)
                action_label = torch.tensor(action_label_dict[action]).repeat(seq_sel1.shape[0]+seq_sel2.shape[0])

                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)
                    action_label_seq = action_label

    # if is not testing or validation then get the data statistics
    # if not (subj == 5 and subj == 11):
    #     data_std = np.std(complete_seq, axis=0)
    #     data_mean = np.mean(complete_seq, axis=0)
    data_std = np.std(complete_seq, axis=0)
    data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0



    return action_label_seq.long(), sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_3d(path_to_dataset, subjects, actions, sample_rate, seq_len):
    """

    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    :param path_to_dataset:
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len:
    :return:
    """

    sampled_seq = []
    complete_seq = []
    for subj in subjects:
        print("Reading subject {0}".format(subj))
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(action_sequence[even_list, :])
                    the_seq = Variable(torch.from_numpy(the_sequence)).float().cuda()
                    # remove global rotation and translation
                    the_seq[:, 0:6] = 0
                    p3d = expmap2xyz_torch(the_seq)
                    the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()

                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames1 = len(even_list)
                the_sequence1 = np.array(action_sequence[even_list, :])
                the_seq1 = Variable(torch.from_numpy(the_sequence1)).float().cuda()
                the_seq1[:, 0:6] = 0
                p3d1 = expmap2xyz_torch(the_seq1)
                the_sequence1 = p3d1.view(num_frames1, -1).cpu().data.numpy()

                # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames2 = len(even_list)
                the_sequence2 = np.array(action_sequence[even_list, :])
                the_seq2 = Variable(torch.from_numpy(the_sequence2)).float().cuda()
                the_seq2[:, 0:6] = 0
                p3d2 = expmap2xyz_torch(the_seq2)
                the_sequence2 = p3d2.view(num_frames2, -1).cpu().data.numpy()

                # print("action:{}".format(action))
                # print("subact1:{}".format(num_frames1))
                # print("subact2:{}".format(num_frames2))
                fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len)
                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel1), axis=0)
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence1, axis=0)
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    return sampled_seq, dimensions_to_ignore, dimensions_to_use


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 180
    T2 = frame_num2 - 180  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2

def hm36_fetch_data_train(folder_list, rank, actions, root_path, sample_rate=2):
    out_poses_3d = {}
    complete_seq = []
    for subj in folder_list:
        if rank <= 0:
            print('load S%d' % subj)
        for action_name in actions:
            for subact in [1, 2]:
                filename = '{0}/S{1}/{2}_{3}.txt'.format(root_path, subj, action_name, subact)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                num_frames = len(even_list)
                the_sequence = np.array(action_sequence[even_list, :])
                the_seq = Variable(torch.from_numpy(the_sequence)).float().cuda()
                # remove global rotation and translation
                the_seq[:, 0:6] = 0
                p3d = expmap2xyz_torch(the_seq)
                the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()
                out_poses_3d[('s%d'%subj, action_name, str(subact))] = the_sequence
                if len(complete_seq) == 0:
                    complete_seq = the_sequence
                else:
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)
    return out_poses_3d, dimensions_to_ignore, dimensions_to_use

def hm36_fetch_data_test(folder_list, rank, actions, root_path, sample_rate=2, out_frame_num=128):
    complete_seq = []
    sampled_seq = []
    for subj in folder_list:
        for action_name in actions:
            if rank <= 0:
                print('load S%d %s' % (subj, action_name))
            filename = '{0}/S{1}/{2}_{3}.txt'.format(root_path, subj, action_name, 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            even_list = range(0, n, sample_rate)

            num_frames1 = len(even_list)
            the_sequence1 = np.array(action_sequence[even_list, :])
            the_seq1 = Variable(torch.from_numpy(the_sequence1)).float().cuda()
            the_seq1[:, 0:6] = 0
            p3d1 = expmap2xyz_torch(the_seq1)
            the_sequence1 = p3d1.view(num_frames1, -1).cpu().data.numpy()

            # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
            filename = '{0}/S{1}/{2}_{3}.txt'.format(root_path, subj, action_name, 2)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            even_list = range(0, n, sample_rate)

            num_frames2 = len(even_list)
            the_sequence2 = np.array(action_sequence[even_list, :])
            the_seq2 = Variable(torch.from_numpy(the_sequence2)).float().cuda()
            the_seq2[:, 0:6] = 0
            p3d2 = expmap2xyz_torch(the_seq2)
            the_sequence2 = p3d2.view(num_frames2, -1).cpu().data.numpy()

            fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, out_frame_num)

            seq_sel1 = the_sequence1[fs_sel1, :]
            seq_sel2 = the_sequence2[fs_sel2, :]
            if len(sampled_seq) == 0:
                sampled_seq = seq_sel1
                sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                complete_seq = the_sequence1
                complete_seq = np.append(complete_seq, the_sequence2, axis=0)
            else:
                sampled_seq = np.concatenate((sampled_seq, seq_sel1), axis=0)
                sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                complete_seq = np.append(complete_seq, the_sequence1, axis=0)
                complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)
    return sampled_seq, dimensions_to_ignore, dimensions_to_use




if __name__ == "__main__":
    r = np.random.rand(2, 3) * 10
    # r = np.array([[0.4, 1.5, -0.0], [0, 0, 1.4]])
    r1 = r[0]
    R1 = expmap2rotmat(r1)
    q1 = rotmat2quat(R1)
    # R1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    e1 = rotmat2euler(R1)

    r2 = r[1]
    R2 = expmap2rotmat(r2)
    q2 = rotmat2quat(R2)
    # R2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    e2 = rotmat2euler(R2)

    r = Variable(torch.from_numpy(r)).cuda().float()
    # q = expmap2quat_torch(r)
    R = expmap2rotmat_torch(r)
    q = rotmat2quat_torch(R)
    # R = Variable(torch.from_numpy(
    #     np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]]))).cuda().float()
    eul = rotmat2euler_torch(R)
    eul = eul.cpu().data.numpy()
    R = R.cpu().data.numpy()
    q = q.cpu().data.numpy()

    if np.max(np.abs(eul[0] - e1)) < 0.000001:
        print('e1 clear')
    else:
        print('e1 error {}'.format(np.max(np.abs(eul[0] - e1))))
    if np.max(np.abs(eul[1] - e2)) < 0.000001:
        print('e2 clear')
    else:
        print('e2 error {}'.format(np.max(np.abs(eul[1] - e2))))

    if np.max(np.abs(R[0] - R1)) < 0.000001:
        print('R1 clear')
    else:
        print('R1 error {}'.format(np.max(np.abs(R[0] - R1))))

    if np.max(np.abs(R[1] - R2)) < 0.000001:
        print('R2 clear')
    else:
        print('R2 error {}'.format(np.max(np.abs(R[1] - R2))))

    if np.max(np.abs(q[0] - q1)) < 0.000001:
        print('q1 clear')
    else:
        print('q1 error {}'.format(np.max(np.abs(q[0] - q1))))

def unNormalizeData_torch(outputs, all_seq, data_mean, data_std, dim_used):#, actions, one_hot
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    outputs: n*t*d matrix with normalized data
    all_seq: n*t_all*d_all
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_used: vector with dimensions used by the model

  Returns
    origData: data originally used to n*t*d_all
  """
  if torch.cuda.is_available():
      data_mean_tensor = torch.tensor(data_mean).cuda()
      data_std_tensor = torch.tensor(data_std).cuda()
  else:
      data_mean_tensor = torch.tensor(data_mean)
      data_std_tensor = torch.tensor(data_std)
  n, t, _ = outputs.shape
  _, _, dim_full_len = all_seq.shape
  d = data_mean.shape[0]
  origData = all_seq.clone()[:, -t:, :]
  origData[:, :, dim_used] = outputs.clone()


  origData = origData.contiguous().view(-1, dim_full_len)
  origData = origData* data_std_tensor +  data_mean_tensor
  ## remove the first 6 elements since it is for global rotation and translation
  origData[:, 0:6] = all_seq.clone()[:, -t:, 0:6].contiguous().view(-1, 6)

  origData = origData.view(n, t, dim_full_len)
  return origData

def cal_mpjpe(poses_diff, mask):
    N,C,T = poses_diff.shape
    mask_index = np.where(mask[0,0]==0)[0]
    poses_diff = poses_diff[:,:,mask_index].reshape([N,-1,3,mask_index.shape[0]])
    errors = np.mean(np.mean(LA.norm(poses_diff, axis=2), axis=1),axis=1)#N
    return errors

def cal_mpjpe_joint(poses_diff, mask):
    N,C,T = poses_diff.shape
    mask1 = (mask==0)
    num_joints = np.sum(mask)//3
    poses_diff = poses_diff*mask1
    errors = np.sum(LA.norm(poses_diff, axis=2))/num_joints
    return errors

def ang2joint(p3d0, pose,
              parent={0: -1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9,
                      15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}):
    """

    :param p3d0:[batch_size, joint_num, 3]
    :param pose:[batch_size, joint_num, 3]
    :param parent:
    :return:
    """
    # model_path = './model.npz'
    # params = np.load(model_path, allow_pickle=True)
    # kintree_table = params['kintree_table']
    batch_num = p3d0.shape[0]
    # id_to_col = {kintree_table[1, i]: i
    #              for i in range(kintree_table.shape[1])}
    # parent = {
    #     i: id_to_col[kintree_table[0, i]]
    #     for i in range(1, kintree_table.shape[1])
    # }
    # parent = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13,
    #           17: 14, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21}
    jnum = len(parent.keys())
    # v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
    # J = torch.matmul(self.J_regressor, v_shaped)
    # face_J = v_shaped[:, [333, 2801, 6261], :]
    J = p3d0
    R_cube_big = rodrigues(pose.contiguous().view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
    results = []
    results.append(
        with_zeros(torch.cat((R_cube_big[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
    )
    # for i in range(1, kintree_table.shape[1]):
    for i in range(1, jnum):
        results.append(
            torch.matmul(
                results[parent[i]],
                with_zeros(
                    torch.cat(
                        (R_cube_big[:, i], torch.reshape(J[:, i, :] - J[:, parent[i], :], (-1, 3, 1))),
                        dim=2
                    )
                )
            )
        )

    stacked = torch.stack(results, dim=1)
    J_transformed = stacked[:, :, :3, 3]
    return J_transformed

def rodrigues(r):
    """
    Rodrigues' rotation formula that turns axis-angle tensor into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size * angle_num, 3, 3].

    """
    eps = r.clone().normal_(std=1e-8)
    theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)
    # theta = torch.norm(r, dim=(1, 2), keepdim=True)  # dim cannot be tuple
    theta_dim = theta.shape[0]
    r_hat = r / theta
    cos = torch.cos(theta)
    z_stick = torch.zeros(theta_dim, dtype=torch.float).to(r.device)
    m = torch.stack(
        (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
         -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
    m = torch.reshape(m, (-1, 3, 3))
    i_cube = (torch.eye(3, dtype=torch.float).unsqueeze(dim=0) \
              + torch.zeros((theta_dim, 3, 3), dtype=torch.float)).to(r.device)
    A = r_hat.permute(0, 2, 1)
    dot = torch.matmul(A, r_hat)
    R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
    return R


def with_zeros(x):
    """
    Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

    Parameter:
    ---------
    x: Tensor to be appended.

    Return:
    ------
    Tensor after appending of shape [4,4]

    """
    ones = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float
    ).expand(x.shape[0], -1, -1).to(x.device)
    ret = torch.cat((x, ones), dim=1)
    return ret


def pack(x):
    """
    Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

    Parameter:
    ----------
    x: A tensor of shape [batch_size, 4, 1]

    Return:
    ------
    A tensor of shape [batch_size, 4, 4] after appending.

    """
    zeros43 = torch.zeros(
        (x.shape[0], x.shape[1], 4, 3), dtype=torch.float).to(x.device)
    ret = torch.cat((zeros43, x), dim=3)
    return ret



