import torch
from .base_model import BaseModel
from . import network1, base_function1, external_function
from util import task_pose
import itertools

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

import random
import torch.nn as nn
import importlib
import viz_hm36
import pdb



class pose_class(BaseModel):
    """This class implements the pose completion"""
    def name(self):
        return "Pose class Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--activation', type=str, default='LeakyReLU', help='activation type')
        parser.add_argument('--init_type', type=str, default='orthogonal', help='network init type')
        parser.add_argument('--use_spect', type=bool, default=True, help='if use spectrum norm')
        parser.add_argument('--use_coord', type=bool, default=False, help='if use positional encoding or not')
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        parser.add_argument('--layer_num', type=int, default=3, help='# of number of the output scale')
        parser.add_argument('--netd_layer', type=int, default=3, help='# of number of the output scale')
        parser.add_argument('--encoder_L', type=int, default=2, help='# of number of encoder L layers')
        parser.add_argument('--small_feature', type=int, default=128, help='# small feature size of the network')
        parser.add_argument('--large_feature', type=int, default=256, help='# large feature size of the network')
        parser.add_argument('--encoder_use_action', type=bool, default=False, help='if use action label into the encoder network')
        parser.add_argument('--use_action', type=bool, default=True, help='if use action label into the network')
        parser.add_argument('--use_spade', type=bool, default=True, help='if use spade into the network')
        parser.add_argument('--use_attn', type=bool, default=True, help='if use attention into the network')
    
        parser.add_argument('--conv_dilation', type=bool, default=False, help='if use dilated convolution')
        parser.add_argument('--dis_norm', type=str, default='none', help='normalization for discriminator')
        parser.add_argument('--dis_use_action', type=bool, default=True, help='if use action in discriminator')
        parser.add_argument('--dis_use_attn', type=bool, default=False, help='if use attention in discriminator')
        parser.add_argument('--enc_last_norm', type=str, default='none', help='normalization method for the last few layers')
        
        parser.add_argument('--add_classifier', type=bool, default=True, help='add classifier branch during training')
        parser.add_argument('--use_one_hot', type=bool, default=False, help='if use one-hot vector for action')
        parser.add_argument('--class_loss', type=str, default='smoothCE', help='which loss function to use {L1|smoothCE}')
        parser.add_argument('--class_one_pred', type=bool, default=False, help='get one prediction result or not')
        parser.add_argument('--final_smooth', type=bool, default=True, help='if use final smoothness or not')

        
        if is_train:

            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths {"one"|"two"|"CVAE"}')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
        return parser

    def __init__(self, opt):
        """Initial the model"""
        BaseModel.__init__(self, opt)
        if opt.reduce_layer:
            opt.output_scale, opt.netd_layer = 2, 2
        self.actions_num = len(self.opt.id2action)
        if opt.use_one_hot:
            self.action_dim = self.actions_num
        else:
            self.action_dim = 300
        self.loss_names = ['kl_rec', 'kl_g', 'app_rec', 'app_g', 'ad_g', 'poses_d', 'ad_rec', 'poses_d_rec']
        self.visual_names = ['poses_m', 'poses_c', 'poses_truth', 'poses_out', 'poses_g', 'poses_rec']
        self.value_names = ['u_m', 'sigma_m', 'u_post', 'sigma_post', 'u_prior', 'sigma_prior']
        self.model_names = ['E', 'G', 'D', 'D_rec']
        self.distribution = []

        # define the inpainting model
        self.net_E = network1.define_e1(input_nc=opt.joint_num*3, norm='none', opt=opt,action_dim=self.action_dim)
        self.net_G = network1.define_g1(output_nc = opt.joint_num*3, L = 0, frame_num = opt.out_frame_num//(2**opt.layer_num), norm = 'instance', action_dim = self.action_dim, opt = opt)

        # define the discriminator model
        self.net_D = network1.define_d1(input_nc=opt.joint_num*3, num_classes=len(opt.id2action), layers=opt.netd_layer,action_dim=self.action_dim, opt=opt)

        self.final_mean_filter = network1.define_mean_filter(kernel_size=5, gpu_ids=opt.gpu_ids, dist=opt.dist, rank=opt.rank)
        if opt.add_classifier:
            self.net_C = network1.define_c1(input_nc=opt.joint_num*3, num_classes=len(opt.id2action), opt=opt)

            self.model_names.append('C')
            self.loss_names.append('class')
            self.loss_names.append('D_class')


        self.net_D_rec = network1.define_d1(input_nc=opt.joint_num*3, num_classes=len(opt.id2action), layers=opt.netd_layer,action_dim=self.action_dim, opt=opt)


        if self.isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.CEloss = torch.nn.CrossEntropyLoss()

            # define the optimizer
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters()),
                                filter(lambda p: p.requires_grad, self.net_E.parameters())), lr=opt.lr,
                betas=(0.0, 0.999))
            if opt.add_classifier:
                self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                    filter(lambda p: p.requires_grad, self.net_D_rec.parameters()),filter(lambda p: p.requires_grad, self.net_C.parameters())),
                                                    lr=opt.lr, betas=(0.0, 0.999))
            else:
                self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters()),
                                                    filter(lambda p: p.requires_grad, self.net_D_rec.parameters())),
                                                    lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulersf
        self.setup(opt)
        if self.opt.save_imgs and self.opt.rank<=0:
            self.img_initalize()
        if self.opt.save_video and self.opt.rank<=0:
            self.video_initalize()




    def set_input(self, input, epoch=0, i = 0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.epoch = epoch
        self.i_data = i

        self.poses_truth = input['rel_poses_3d'].clone() # n,c, t
        self.poses_truth_all = input['rel_poses_3d'].clone()  # n,c, t
        n, _, t = self.poses_truth.shape
        self.mask = input['mask'].clone().view(n, -1, t)# n,1, t
        self.actionmap = input['action_vec'].clone()
        self.action_id = input['action_id'].clone()
        self.action_name = input['action_name']




        if len(self.gpu_ids) > 0:
            self.poses_truth = self.poses_truth.cuda()
            self.mask = self.mask.cuda()
            self.actionmap = self.actionmap.cuda()
            self.action_id = self.action_id.cuda()
            self.poses_truth_all = self.poses_truth_all.cuda()

        # get I_m and I_c for image with mask and complement regions for training
        self.poses_m = self.mask * self.poses_truth
        self.poses_c = (1 - self.mask) * self.poses_truth

        # get multiple scales image ground truth and mask for training
        self.scale_poses = task_pose.scale_pyramid(self.poses_truth, self.opt.output_scale)
        self.scale_mask = task_pose.scale_pyramid(self.mask, self.opt.output_scale)

    def test(self, get_all_samplings=False):
        """Forward function used in test time"""


        # encoder process
        distribution, f = self.net_E(self.poses_m, actionmap=self.actionmap)
        q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1])

        # decoder process
        # for saving results
        self.vis_row = random.randint(0, self.poses_truth.shape[0]-1)
        if get_all_samplings or (self.i_data % self.opt.test_save_freq == 0 and (self.opt.save_imgs or self.opt.save_video)):
            nsampling = self.opt.nsampling
        else:
            nsampling = 1
  
        for i in range(nsampling):
            z = q_distribution.sample()
            f_e = f

            scale_mask = task_pose.scale_img(self.mask, size=[f_e[-1].size(2)])
            self.poses_g, attn = self.net_G(z, f_m=f[-1], f_e=f, mask=scale_mask.chunk(3, dim=1)[0],
                                            actionmap=self.actionmap)
        
            self.poses_out = (1 - self.mask) * self.poses_g[-1].detach() + self.mask * self.poses_m
            if self.opt.final_smooth:
                self.poses_out = (1 - self.mask) *self.final_mean_filter(self.poses_out) + self.mask *self.poses_m

            self.score = self.net_D(self.poses_out, actionmap=self.actionmap)

            if get_all_samplings:
                if i == 0:
                    self.all_samplings_gen = self.poses_out.clone().unsqueeze(0)
                else:
                    self.all_samplings_gen = torch.cat((self.all_samplings_gen, self.poses_out.clone().unsqueeze(0)),dim=0) # K, N, T, C
 
            input_3D_plot, pred_3D_plot, gt_3D_plot = self.prepare_plot_results()
      

            if self.opt.save_imgs and (self.i_data % self.opt.test_save_freq == 0):
                self.save_imgs(input_3D_plot, pred_3D_plot, gt_3D_plot, n_sample=i, row=self.vis_row, rank=self.opt.rank)
            if self.opt.save_video and (self.i_data % self.opt.test_save_freq == 0):
                    self.save_video(input_3D_plot, pred_3D_plot, gt_3D_plot, n_sample=i, row=self.vis_row, rank=self.opt.rank)


    def prepare_plot_results(self):
        input_3D = self.poses_truth_all.clone().permute(0,2,1).detach().cpu().numpy()*1000 #  n,t, c
        pred_3D = self.poses_truth_all.clone().permute(0, 2, 1).detach().cpu().numpy()*1000 # n,t, c
        gt_3D = self.poses_truth_all.clone().permute(0, 2, 1).detach().cpu().numpy()*1000 # n,t, c


        input_3D = (self.mask * self.poses_truth_all).clone().permute(0, 2,1).detach().cpu().numpy() * 1000  # n,t, c
        pred_3D = self.poses_out.detach().permute(0, 2, 1).cpu().numpy() * 1000
        return input_3D, pred_3D, gt_3D

    def get_distribution(self, distributions):
        """Calculate encoder distribution for img_m, img_c"""
        # get distribution
        sum_valid = (torch.mean(self.mask.view(self.mask.size(0), -1), dim=1) - 1e-5).view(-1, 1, 1)
        m_sigma = 1 / (1 + ((sum_valid - 0.8) * 8).exp_())
        p_distribution, q_distribution, kl_rec, kl_g = 0, 0, 0, 0
        self.distribution = []
        for distribution in distributions:
            p_mu, p_sigma, q_mu, q_sigma = distribution
            # the assumption distribution for different mask regions
            m_distribution = torch.distributions.Normal(torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma))
    
            # the post distribution from mask regions
            p_distribution = torch.distributions.Normal(p_mu, p_sigma)
            p_distribution_fix = torch.distributions.Normal(p_mu.detach(), p_sigma.detach())
            # the prior distribution from valid region
            q_distribution = torch.distributions.Normal(q_mu, q_sigma)

            # kl divergence
            kl_rec += torch.distributions.kl_divergence(m_distribution, p_distribution)
            if self.opt.train_paths == "one":
                kl_g += torch.distributions.kl_divergence(m_distribution, q_distribution)
            elif self.opt.train_paths == "two" or self.opt.train_paths == "CVAE":
                kl_g += torch.distributions.kl_divergence(p_distribution_fix, q_distribution)
            self.distribution.append([torch.zeros_like(p_mu), m_sigma * torch.ones_like(p_sigma), p_mu, p_sigma, q_mu, q_sigma])

        return p_distribution, q_distribution, kl_rec, kl_g

    def get_G_inputs(self, p_distribution, q_distribution, f):
        """Process the encoder feature and distributions for generation network"""
        f_m = torch.cat([f[-1].chunk(2)[0], f[-1].chunk(2)[0]], dim=0)

        f_e = []
        for i in range(len(f)):
            f_e.append(torch.cat([f[i].chunk(2)[0], f[i].chunk(2)[0]], dim=0))

        scale_mask = task_pose.scale_img(self.mask, size=[f_e[-1].size(2)])
        mask = torch.cat([scale_mask.chunk(3, dim=1)[0], scale_mask.chunk(3, dim=1)[0]], dim=0)
        z_p = p_distribution.rsample()
        z_q = q_distribution.rsample()
        z = torch.cat([z_p, z_q], dim=0)
        return z, f_m, f_e, mask

    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        distributions, f = self.net_E(self.poses_m, self.poses_c, actionmap=self.actionmap)
        p_distribution, q_distribution, self.kl_rec, self.kl_g = self.get_distribution(distributions)

        # decoder process
        z, f_m, f_e, mask = self.get_G_inputs(p_distribution, q_distribution, f)
        results, attn = self.net_G(z, f_m, f_e, mask, actionmap=self.actionmap)
        self.poses_rec = []
        self.poses_g = []
        for result in results:
            poses_rec, poses_g = result.chunk(2)
            self.poses_rec.append(poses_rec)
            self.poses_g.append(poses_g)
        self.poses_out = (1-self.mask) * self.poses_g[-1].detach() + self.mask * self.poses_truth

        # save if needed
        input_3D_plot, pred_3D_plot, gt_3D_plot = self.prepare_plot_results()
        if self.i_data % self.opt.display_freq == 0 and self.opt.rank<=0:
            self.vis_row = random.randint(0, self.poses_truth.shape[0]-1)
            if self.opt.save_imgs :
                self.save_imgs(input_3D_plot, pred_3D_plot, gt_3D_plot,row=self.vis_row, rank=self.opt.rank)
            if self.opt.save_video:
                self.save_video(input_3D_plot, pred_3D_plot, gt_3D_plot,row=self.vis_row, rank=self.opt.rank)

    def backward_D_basic(self, netD, real, fake, actionmap=None):
        """Calculate GAN loss for the discriminator"""

        D_real = netD(real, actionmap)
        D_fake = netD(fake.detach(), actionmap)
        loss_D_class = torch.tensor(0).cuda()
        D_real_loss = self.GANloss(D_real, True, True)
        D_fake_loss = self.GANloss(D_fake, False, True)
        # loss for discriminator
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss +=gradient_penalty


        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function1._unfreeze(self.net_D, self.net_D_rec)
        self.loss_poses_d = self.backward_D_basic(self.net_D, self.poses_truth, self.poses_g[-1], actionmap=self.actionmap)
        self.loss_poses_d_rec = self.backward_D_basic(self.net_D_rec, self.poses_truth, self.poses_rec[-1], actionmap=self.actionmap)

        if self.opt.add_classifier:
            base_function1._unfreeze(self.net_C)
            class_pred_real = self.net_C(self.poses_truth, self.opt.class_one_pred)
            self.loss_D_class = self.CEloss(class_pred_real, self.action_id.view(-1,1).repeat(1,class_pred_real.size(-1)))
        else:
            self.loss_D_class = torch.tensor(0).cuda()

        if self.opt.train_paths == 'one' or self.opt.train_paths == 'CVAE':
            self.loss_poses_d_rec = self.loss_poses_d_rec * 0
        loss = self.loss_poses_d + self.loss_poses_d_rec + self.loss_D_class
        loss.backward()


    def backward_G(self):
        """Calculate training loss for the generator"""

        # encoder kl loss
        self.loss_kl_rec = self.kl_rec.mean() * self.opt.lambda_kl * self.opt.output_scale
        self.loss_kl_g = self.kl_g.mean() * self.opt.lambda_kl * self.opt.output_scale

        # generator adversarial loss
        base_function1._freeze(self.net_D, self.net_D_rec)
        if self.opt.add_classifier:
            base_function1._freeze(self.net_C)
        # g loss fake
        if self.opt.add_classifier:
            class_pred_fake = self.net_C(self.poses_g[-1], self.opt.class_one_pred)
            class_pred_fake_rec = self.net_C(self.poses_rec[-1], self.opt.class_one_pred)
            class_pred_real = self.net_C(self.poses_truth, self.opt.class_one_pred)
            if self.opt.class_loss == 'L1':
                softmax = nn.Softmax(dim=1)
                self.loss_class = self.L1loss(softmax(class_pred_fake), softmax(class_pred_real)) + self.L1loss(
                    softmax(class_pred_fake_rec), softmax(class_pred_real))
            elif self.opt.class_loss == 'smoothCE':
                self.loss_class = (external_function.softmax_and_cross_entropy(class_pred_fake, class_pred_real) + external_function.softmax_and_cross_entropy(class_pred_fake_rec,class_pred_real))

        else:
            self.loss_class = torch.tensor(0).cuda()
        D_fake = self.net_D(self.poses_g[-1], actionmap=self.actionmap)
        D_fake_rec = self.net_D_rec(self.poses_rec[-1], actionmap=self.actionmap)
        D_real = self.net_D_rec(self.poses_truth, actionmap=self.actionmap)


        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g
        self.loss_ad_rec = self.L2loss(D_fake_rec, D_real) * self.opt.lambda_g

        # calculate l1 loss ofr multi-scale outputs
        loss_app_rec, loss_app_g = 0, 0
        for i, (pose_rec_i, pose_fake_i, pose_real_i, mask_i) in enumerate(zip(self.poses_rec, self.poses_g, self.scale_poses, self.scale_mask)):
            if i > self.opt.rec_cal_layer:
                loss_app_rec += self.L1loss(pose_rec_i, pose_real_i)
                if self.opt.train_paths == "one" or self.opt.train_paths == "CVAE":
                    loss_app_g += self.L1loss(pose_fake_i, pose_real_i)
                elif self.opt.train_paths == "two":
                    loss_app_g += self.L1loss(pose_fake_i*mask_i, pose_real_i*mask_i)


        self.loss_app_rec = loss_app_rec * self.opt.lambda_rec
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # if one path during the training, just calculate the loss for generation path
        if self.opt.train_paths == "CVAE" :
            self.loss_app_rec = self.loss_app_rec * 0
            self.loss_ad_rec = self.loss_ad_rec * 0
        elif self.opt.train_paths == "one":
            self.loss_kl_rec = self.loss_kl_rec * 0
            self.loss_app_rec = self.loss_app_rec * 0
            self.loss_ad_rec = self.loss_ad_rec * 0

        total_loss = 0

        for name in self.loss_names:
            if name != 'poses_d' and name != 'poses_d_rec' and name != 'D_class':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the image completion results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def img_initalize(self):
        if self.opt.dataset_name == 'hm36':
            viz = viz_hm36

        self.vis_frames = (self.opt.out_frame_num//self.opt.num_img_persample)//self.opt.vis_stride
        self.fig = plt.figure(figsize=(self.opt.save_img_w, self.opt.save_img_h)) if hasattr(self.opt, 'save_img_w') and hasattr(self.opt, 'save_img_h') else plt.figure(figsize=(30, 10))#30,10
        self.fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        self.ax = [[], [], []]
        self.ob = [[], [], []]
        for k in range(self.vis_frames):
            for j in range(3):
                self.ax[j].append(
                    self.fig.add_subplot(3, self.vis_frames, self.vis_frames * j + k + 1))  # , projection='3d'))
                self. ob[j].append(viz.Ax2DPose(self.ax[j][k]))

    def video_initalize(self):
        # return None
        if self.opt.dataset_name == 'hm36':
            viz = viz_hm36

        self.fig_video = plt.figure(figsize=(7, 3))#(7,3)
        self.fig_video.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        self.ob_video = viz.Ax3DPose(self.fig_video, num_obj=3, titles=['input', 'gt', 'pred'])



    def save_imgs(self,input_3D_plot, pred_3D_plot, gt_3D_plot, n_sample=None,row=0, rank=0, id=None, vis_stride=1):
        if hasattr(self, 'action_id'):
            action_name = self.opt.id2action[self.action_id[row].item()]
            if hasattr(self, 'fake_action_id'):
                fake_action_name = self.opt.id2action[self.fake_action_id[row].item()]
                suptitle = 'origin: '+action_name+'  now: '+fake_action_name
                if id is not None:
                    suptitle += ' id: %d'%id
                self.fig.suptitle(suptitle)
            else:
                self.fig.suptitle(action_name)

        if hasattr(self, 'action_id'):
            action_name = self.opt.id2action[self.action_id[row].item()]
            path = os.path.join(self.opt.save_img_dir, str(action_name))
        else:
            path = self.opt.save_img_dir
        # N, C, T = self.gen_poses.shape
        input_3D = input_3D_plot[row]  # N, t, c
        pred_3D = pred_3D_plot[row] # N, t, c
        gt_3D = gt_3D_plot[row]# N, t, c

        ## draw images
        for k in range(self.opt.num_img_persample):
            for t in range(self.vis_frames):
                col = k*self.vis_frames + t*self.opt.vis_stride
                mask = self.mask[row, :, col].view(self.opt.joint_num, 3)
                self.ob[0][t].update(input_3D[col],mask=mask)
                self.ob[1][t].update(gt_3D[col])#,"#8e8e8e", "#383838")
                self.ob[2][t].update(pred_3D[col])  # ,"#3498db","#3498db")

            ## save images
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except:
                    pass

            if n_sample is None:
                fig_name = "%s/%04d_%d" % (path, self.i_data, k)
            elif rank >= 0:
                fig_name = "%s/%04d_sample%d_%d_rank%d" % (path, self.i_data, n_sample, k, rank)
            else:
                fig_name = "%s/%04d_sample%d_%d" % (path, self.i_data, n_sample, k)
            if hasattr(self, 'fake_action_id'):
                fake_action_name = self.opt.id2action[self.fake_action_id[row].item()]
                fig_name = fig_name+fake_action_name
            if id is not None:
                fig_name += '_id%d.png' % id
            else:
                fig_name += '.png'

            # self.fig.savefig('try.png', bbox_inches='tight')
            self.fig.savefig(fig_name, bbox_inches='tight')


    def render_animation(self,input_3D_plot, pred_3D_plot, gt_3D_plot, row, id=None):

        if hasattr(self, 'action_id'):
            action_name = self.opt.id2action[self.action_id[row].item()]
            if hasattr(self, 'fake_action_id'):
                fake_action_name = self.opt.id2action[self.fake_action_id[row].item()]
                subtitle = 'origin: '+action_name+'  now: '+fake_action_name
                if id is not None:
                    subtitle += ' id: %d' %id
                self.fig_video.suptitle(subtitle)
            else:
                self.fig_video.suptitle(action_name)
        T = self.poses_truth.shape[-1]
        pred_lcolor = "#8e8e8e"
        pred_rcolor = "#383838"
        lcolor = ["#3498db","#3498db","#3498db"]
        rcolor = ["#e74c3c","#e74c3c","#e74c3c"]
        input_3D = input_3D_plot[row] # N, t, c
        pred_3D = pred_3D_plot[row] # N, t, c
        gt_3D = gt_3D_plot[row]# N, t, c

        poses = [input_3D, gt_3D, pred_3D]

        def update_video(t):
            if torch.sum((self.mask[row,:,t] == 0))>0:
                lcolor[-1], rcolor[-1] = pred_lcolor, pred_rcolor
            mask = self.mask[row, :, t].view(self.opt.joint_num, 3)
            poses_each = [pose[t] for pose in poses]
            self.ob_video.update(poses_each,mask, lcolor, rcolor)

        anim = FuncAnimation(self.fig_video, update_video, frames=np.arange(0, T), interval=1000 / self.opt.fps, repeat=False)
        return anim





    def save_video(self,input_3D_plot, pred_3D_plot, gt_3D_plot, n_sample=None,row=0, rank=0,id=None):

        anim = self.render_animation(input_3D_plot, pred_3D_plot, gt_3D_plot,row,id)
        if hasattr(self, 'action_id'):
            action_name = self.opt.id2action[self.action_id[row].item()]
            path = os.path.join(self.opt.save_video_dir, str(action_name))
        else:
            path = self.opt.save_video_dir

        # save video
        if (not os.path.exists(path)):
            try:
                os.makedirs(path)
            except:
                pass
        if n_sample is None:
            video_name = '%04d'%self.i_data
        elif rank >=0 :
            video_name = '%04d_%d_rank%d'%(self.i_data, n_sample, rank)
        else:
            video_name = '%04d_%d' % (self.i_data, n_sample)

        if id is not None:
            video_name += '_id%d.mp4'%id
        else:
            video_name += '.mp4'

        if hasattr(self, 'fake_action_id'):
            fake_action_name = self.opt.id2action[self.fake_action_id[row].item()]
            video_name = fake_action_name + video_name

        # anim.save('try.mp4')
        anim.save(os.path.join(path, video_name))
