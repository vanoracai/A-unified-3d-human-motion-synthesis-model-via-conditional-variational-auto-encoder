import numpy as np


class ChunkedGenerator:
    """ refined from https://github.com/facebookresearch/VideoPose3D
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:

    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled

    # my own change:
    stride: sample rate of the whole sequence.
    chosed_num: the number of pairs we choose for testing (when only part of the test are used in testing, this is for
    one start with different actions)
    """

    def __init__(self, batch_size, poses_3d, action_labels, poses_2d=None, cameras=None,
                 pad=0, pad_action = 0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,
                 endless=False, out_frame_num = 1,
                 stride=1, start_stride=1, choosed_num=-1,start_chosed_frame = 0):
        assert poses_3d is not None
        if action_labels is not None:
            assert len(poses_3d) == len(action_labels)
        if poses_2d is not None:
            assert len(poses_3d) == len(poses_2d)


        # Build lineage info
        pairs = []
        self.saved_index = {}
        self.stride = stride
        start_index = 0
        sample_length = (out_frame_num -1 ) * stride +1



        for key in poses_3d.keys():
            assert poses_3d is not None


            # get start and end of the boundary
            if poses_3d[key].shape[0] - (sample_length-1) <= 0:
                print('skip %s/ %s/ %s' %(key[0], key[1], key[2]))
                continue
            bounds = np.arange(poses_3d[key].shape[0] - (sample_length-1))
            assert len(bounds) > 0
            bounds_start = bounds[start_chosed_frame::start_stride]
            if choosed_num != -1:
                bounds_start = bounds[start_chosed_frame:choosed_num]
            bounds_end = bounds_start + sample_length
            n_chunk  = bounds_start.shape[0]

            augment_vector = np.full(n_chunk, False, dtype=bool)
            reverse_augment_vector = np.full(n_chunk, False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1,len(key)]),(n_chunk,1))
            pairs += list(zip(keys, bounds_start, bounds_end, augment_vector,reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds_start, bounds_end, augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds_start, bounds_end, ~augment_vector,~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds_start, bounds_end, ~augment_vector, reverse_augment_vector))
            # for save key index
            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index,end_index]
            start_index = start_index + poses_3d[key].shape[0]

        # Initialize buffers


        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.pad_action = pad_action
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None


        self.poses_3d = poses_3d
        if action_labels is not None:
            self.action_labels = action_labels
        else:
            self.action_labels = None
        if poses_2d is not None:
            self.poses_2d = poses_2d
        else:
            self.poses_2d = None

        self.augment = augment


    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment



    def get_batch_sample(self, seq_i, start, end, flip, reverse):
        # stride: downsample rate, example: downsample=2, get every 2 frame,
        # while in our case, downsample means the skip between two frames, we will still take all frames in the dataset,
        # example: [1,3,5,7] [2,4,6,8] will be set as two subsets, both will be used when training/testing
        data = {}

        stride = self.stride
        seq_name =  tuple(seq_i.tolist())

        # get start and end id
        start_3d = start - self.pad*stride - self.causal_shift
        end_3d = end + self.pad*stride - self.causal_shift

        # get sequential poses/actions
        seq_3d = self.poses_3d[seq_name].copy()
        if self.action_labels is not None:
            seq_action = self.action_labels[seq_name].copy()
        if self.poses_2d is not None:
            seq_2d = self.poses_2d[seq_name].copy()

        # calculate id
        if start_3d < 0:
            low_3d = start_3d%stride
        else:
            low_3d = start_3d
        if end_3d > seq_3d.shape[0]:
            high_3d = seq_3d.shape[0] - (stride - (end_3d - seq_3d.shape[0])%stride)
        else:
            high_3d = end_3d

        pad_left_3d = (low_3d - start_3d)//stride
        pad_right_3d = (end_3d - high_3d)//stride

        # select corresponding poses 3d/2d/action_labels
        if pad_left_3d != 0 or pad_right_3d != 0:
            data['poses_3d'] = np.pad(seq_3d[low_3d:high_3d:stride], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)),
                                      'edge')
            if self.action_labels is not None:
                data['actions'] = np.pad(seq_action[low_3d:high_3d:stride],
                                           ((pad_left_3d, pad_right_3d)), 'edge')
            if self.poses_2d is not None:
                data['poses_2d'] = np.pad(seq_2d[low_3d:high_3d:stride],
                                           ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')

        else:
            data['poses_3d'] = seq_3d[low_3d:high_3d:stride]
            if self.action_labels is not None:
                data['actions'] = seq_action[low_3d:high_3d:stride]
            if self.poses_2d is not None:
                data['poses_2d'] = seq_2d[low_3d:high_3d:stride]



        return data

 
