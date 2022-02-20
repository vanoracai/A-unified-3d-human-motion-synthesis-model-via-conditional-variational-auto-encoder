import argparse
import os
import torch
import model
from util import util
import yaml



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        # base define
        parser.add_argument('--config',  default='', help='configuations of parameters')
        parser.add_argument('--dataset_name', type=str, default='hm36', help='if specified, do distributed multi-gpu task')
        parser.add_argument('--hm36_root_path', type=str, default='./data/hm36', help='path of the dataset')    
        parser.add_argument('--dist', action='store_true', help='if specified, do distributed multi-gpu task')
        parser.add_argument('--debug', type=bool, default=False, help='if debug or not.')
        parser.add_argument('--name', type=str, default='', help='name of the experiment.')                                     
        parser.add_argument('--load_model_name', type=str, default='', help='name of the model loaded.')
        parser.add_argument('--checkpoints_dir', type=str, default='./saved_files/checkpoints', help='models are save here')
        parser.add_argument('--which_iter', type=str, default='latest', help='which iterations to load')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0, 1, 2 use -1 for CPU')
        parser.add_argument('--local_rank', type=int, default=0)
       
        # network options
        parser.add_argument('--reduce_layer', action='store_true', help='if use reduced layer')
        parser.add_argument('--model', type=str, default='pose_class', help='name of the model type. ')
        parser.add_argument('--mask_type', type=int, default=[1, 2, 3, 4],
                    help='mask type, 1: prediction, 2: compeletion(random_consecutive_mask), 3:random_discrete_mask,4:center mask'
                    'eg. [0,1,2,3]')
        parser.add_argument('--mask_weights', type=int, default=[1,1,1,1],
                            help='corresponding weights for each mask type, mask type, 1: prediction, 2: compeletion(random_consecutive_mask), 3:random_discrete_mask,4:center mask'
                            '')
        parser.add_argument('--test_mask_type', type=int, default=[1,2,3,4],
                            help='mask type, 0:geneation, 1: prediction, 2: compeletion(random_consecutive_mask), 3:sparse mask, 4: center_mask (masked the centered ...frames)'
                            'eg. [0,1,2,3]')#
        parser.add_argument('--test_mask_weights', type=int, default=[1,1,1,1],
                            help='corresponding weights for each mask type, mask type, 0:geneation, 1: prediction, 2: compeletion(random_consecutive_mask), 3:sparse mask, 4: center_mask  '
                            '')
        parser.add_argument('--test_interpolation_stride', type=int, default=16,
                            help='test interpolation stride ''')
        parser.add_argument('--rec_cal_layer', type=int, default=-1, help=' layers start to calculate reconstruction loss')
       
        # data pattern define
        parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
        parser.add_argument('--test_batchSize', type=int, default=256, help='input test batch size')
        parser.add_argument('--nsampling', type=int, default=1, help='ramplimg # times for each images')
        parser.add_argument('--nThreads', type=int, default=6, help='# threads for loading data')
        parser.add_argument('--stride', type=int, default=5, help='output frames each time')
        parser.add_argument('--subset', type=bool, default=False, help='use subset dataset for debug')
        parser.add_argument('--num_masked_frames', type=int, default=40, help='length of center consecutive mask')
        parser.add_argument('--train_out_frame_num', type=int, default=128, help='output frames each time')
        parser.add_argument('--test_out_frame_num', type=int, default=128, help='output frames each time')
        parser.add_argument('--joint_num', type=int, default=17, help='number of joints for each frame')
        parser.add_argument('--choosed_num', type=int, default=-1, help='number of pairs to be choosed for each sequence, default:-1, do not change the original pairs')        
        parser.add_argument('--train_start_stride', type=int, default=1, help='stride of pairs')
        parser.add_argument('--test_start_stride', type=int, default=8, help='stride of pairs')
        parser.add_argument('--fixed_input_num', type=int, default=10, help='input num for generation')
        parser.add_argument('--actions_filter', type=str, default='all', help='which kind of actions are used {all|distinguished}')
        parser.add_argument('--start_chosed_frame', type=int, default=0, help='start frame of testing frames')
       
        # save options
        parser.add_argument('--test_save_freq', type=int, default=2, help='frequency of saving test results')
        parser.add_argument('--save_imgs', type=bool, default=True, help='save imgs or not during training/testinng')
        parser.add_argument('--save_video', type=bool, default=True, help='save videos or not during training/testinng')
        parser.add_argument('--save_poses', type=bool, default=False, help='save poses during testinng')
        parser.add_argument('--save_video_dir', type=str, default='save_video', help='save videos')
        parser.add_argument('--fps', type=int, default=20, help='fps for video display')
        parser.add_argument('--num_img_persample', type=int, default=4, help='how many imgs to save for one sample, eg. 4, save 4 images for 1 result')
        parser.add_argument('--vis_stride', type=int, default=1, help='visualization stride of saved img, eg. 2 means draw a pose every two frames')


        return parser

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            parser = self.initialize(self.parser)

        # get basic options
        opt, _ = parser.parse_known_args()

        # check configuation file
        if opt.config is not None:
            with open(opt.config, 'r') as f:
                default_arg = yaml.safe_load(f)

            if 'model' in default_arg:
                opt.model = default_arg['model']

        # modify the options for different models
        if isinstance(opt.model, list):
            # contain more than two models
            for model_name in opt.model:
                model_option_set = model.get_option_setter(model_name)
                parser = model_option_set(parser, self.isTrain)
        else:
            model_option_set = model.get_option_setter(opt.model)
            parser = model_option_set(parser, self.isTrain)
        if opt.config is not None:
            parser.set_defaults(**default_arg)
        opt = parser.parse_args()

        return opt

    def parse(self):
        """Parse the options"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain



        if opt.debug:
            opt.nThreads = 0
            opt.subset = True
            opt.save_imgs = False
            opt.save_video = False

        # for single gpu work, use shell CUDA_VISIBLE_DEVICES to control
        if not opt.dist:
            opt.gpu_ids = '0'


        if opt.dataset_name == 'hm36':
            opt.joint_num = 17



        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids):
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt

        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(expr_dir):
           util.mkdirs(expr_dir)

        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')


