from .base_function1 import *
from .external_function import SpectralNorm
import torch.nn.functional as F

import numpy as np


##############################################################################################################
# Network function
##############################################################################################################


def define_e1(input_nc=3, norm='none', opt=None, concat=False,action_dim=300):
    net = ResEncoder(input_nc, ngf=opt.small_feature, z_nc=opt.large_feature, img_f=opt.large_feature, L=opt.encoder_L, layers=opt.layer_num, norm=norm, activation=opt.activation, use_spect=opt.use_spect, use_coord=opt.use_coord, concat=concat, frame_num= opt.out_frame_num, action_dim=action_dim, use_action=opt.encoder_use_action, conv_dilation=opt.conv_dilation, enc_last_norm=opt.enc_last_norm, use_spade=opt.use_spade)

    return init_net(net, opt.init_type, opt.activation, opt.gpu_ids, opt.dist,opt.rank)


def define_g1(output_nc=3, L=0, frame_num =16,norm='instance',action_dim=300,opt=None,concat=False):

    net = ResGenerator1(output_nc, ngf=opt.small_feature, z_nc=opt.large_feature, img_f=opt.large_feature, L=L, layers=opt.layer_num,
                        norm=norm, activation=opt.activation, output_scale=opt.output_scale, use_spect=opt.use_spect, use_coord=opt.use_coord,
                        use_attn=opt.use_attn, concat=concat, frame_num=frame_num, use_spade = opt.use_spade, action_dim=action_dim,
                        use_action = opt.use_action, conv_dilation=opt.conv_dilation,  use_one_hot=opt.use_one_hot)
    return init_net(net, opt.init_type, opt.activation, opt.gpu_ids, opt.dist,opt.rank)


 

def define_d1(input_nc = 3,num_classes=15, layers=3,action_dim=300,opt=None,concat=False):

    net = ResDiscriminator(input_nc, ndf=opt.small_feature, img_f=opt.large_feature, layers=layers, norm=opt.dis_norm, activation=opt.activation, use_spect=opt.use_spect, use_coord=opt.use_coord, use_attn=opt.dis_use_attn, concat=concat, frame_num=opt.out_frame_num, action_dim=action_dim, use_action = opt.dis_use_action, conv_dilation=opt.conv_dilation, use_one_hot=opt.use_one_hot, num_classes=num_classes)

    return init_net(net, opt.init_type, opt.activation, opt.gpu_ids, opt.dist,opt.rank)



def define_c1(input_nc, num_classes=15, opt=None, concat=False):

    net = ResClassifier(input_nc, ndf=opt.small_feature, img_f=opt.large_feature, layers=opt.netd_layer, norm=opt.dis_norm, activation=opt.activation, use_spect=opt.use_spect, use_coord=opt.use_coord, use_attn=opt.dis_use_attn, concat=concat,
                               frame_num = opt.out_frame_num, conv_dilation=opt.conv_dilation, num_classes=num_classes)
    return init_net(net, opt.init_type, opt.activation, opt.gpu_ids, opt.dist, opt.rank)



def define_mean_filter(kernel_size=10, init_type='orthogonal', activation='ReLU', gpu_ids=[], dist=False, rank=-1):
    net = nn.Sequential(nn.ReflectionPad1d(int(kernel_size / 2)),
                                               nn.AvgPool1d(kernel_size=kernel_size, stride=1))
    return net
#############################################################################################################
# Network structure
#############################################################################################################
class ResEncoder(nn.Module):
    """
    ResNet Encoder Network
    :param input_nc: number of channels in input
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ngf=64, z_nc=128, img_f=1024, L=6, layers=6, norm='none', activation='ReLU',
                 use_spect=True, use_coord=False, concat=False, frame_num=5000, action_dim=300, use_action=False, conv_dilation=False, enc_last_norm='none',use_spade=False):
        super(ResEncoder, self).__init__()

        self.layers = layers
        self.z_nc = z_nc
        self.L = L
        self.use_action = use_action
        max_len = frame_num

        if self.use_action:
            self.conv_action = nn.Conv1d(action_dim, input_nc, 1)
            input_nc *= 2

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ngf, norm_layer, nonlinearity, use_spect, use_coord, concat, max_len, conv_dilation)
        max_len = max_len//2
        mult = 1
        for i in range(layers-1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ngf)
            block = ResBlock(ngf * mult_prev, ngf * mult, ngf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord, concat, max_len, conv_dilation)
            setattr(self, 'encoder' + str(i), block)
            max_len = max_len//2

        # inference part
        last_norm_layer = get_norm_layer(norm_type=enc_last_norm)
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf *mult, last_norm_layer, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation=False, use_action=use_action, use_spade=use_spade, action_dim=action_dim)
            setattr(self, 'infer_prior' + str(i), block)

        self.posterior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, last_norm_layer, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation=False, use_action=use_action, use_spade=use_spade, action_dim=action_dim)
        self.prior = ResBlock(ngf * mult, 2*z_nc, ngf * mult, last_norm_layer, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation=False, use_action=use_action, use_spade=use_spade, action_dim=action_dim)

    def forward(self, img_m, img_c=None, actionmap=None):
        """
        :param img_m: image with mask regions I_m
        :param img_c: complement of I_m, the mask regions
        :return distribution: distribution of mask regions, for training we have two paths, testing one path
        :return feature: the conditional feature f_m, and the previous f_pre for auto context attention
        """
        # process actionmap
        if actionmap is not None and self.use_action:
            actionmap_conv = self.conv_action(actionmap)
            img_m = torch.cat((img_m, actionmap_conv), 1)
            if type(img_c) != type(None):
                img_c = torch.cat((img_c, actionmap_conv), 1)

        if type(img_c) != type(None):
            img = torch.cat([img_m, img_c], dim=0)
        else:
            img = img_m

        # encoder part
        out = self.block0(img)
        feature = [out]
        for i in range(self.layers-1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            feature.append(out)

        # infer part
        # during the training, we have two paths, during the testing, we only have one paths
        if type(img_c) != type(None):
            distribution = self.two_paths(out, actionmap)
            return distribution, feature
        else:
            distribution = self.one_path(out, actionmap)
            return distribution, feature

    def one_path(self, f_in, actionmap=None):
        """one path for baseline training or testing"""
        f_m = f_in
        distribution = []

        # infer state
        for i in range(self.L):
            infer_prior = getattr(self, 'infer_prior' + str(i))
            f_m = infer_prior(f_m, actionmap)

        # get distribution
        o = self.prior(f_m, actionmap)
        q_mu, q_std = torch.split(o, self.z_nc, dim=1)
        distribution.append([q_mu, F.softplus(q_std)])

        return distribution

    def two_paths(self, f_in, actionmap=None):
        """two paths for the training"""
        f_m, f_c = f_in.chunk(2)
        distributions = []

        # get distribution
        o = self.posterior(f_c, actionmap)
        p_mu, p_std = torch.split(o, self.z_nc, dim=1)
        distribution = self.one_path(f_m, actionmap)
        distributions.append([p_mu, F.softplus(p_std), distribution[0][0], distribution[0][1]])

        return distributions

class ResGenerator1(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True,concat=False,frame_num=5000, use_spade=False, action_dim=300, use_action=False, conv_dilation=False, reduce_layer=False, use_one_hot=False):
        super(ResGenerator1, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.use_action = use_action
        max_len = frame_num

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord,
                                  concat, max_len, conv_dilation, use_action, use_spade, action_dim, use_one_hot)
        # self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation, use_action, use_spade, action_dim)
        if use_attn:
            self.attn_g = Attn_Net(ngf * mult, None, use_spect)
        self.outconv_g = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord, concat, max_len)

        # set mean convoution
        kernel_size = 5
        self.mean_filter = nn.Sequential(nn.ReflectionPad1d(int(kernel_size / 2)),
                                               nn.AvgPool1d(kernel_size=kernel_size, stride=1))

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation, use_action, use_spade, action_dim)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):

            mult_prev = mult
            mult = max(min(2 ** (layers - i - 2), img_f // ngf),2**0)
            if i > layers - output_scale:
                
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_spade, action_dim, max_len, use_action, use_one_hot)
            else:
                
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_spade, action_dim, max_len, use_action, use_one_hot)
            max_len = max_len * 2
            setattr(self, 'decoder' + str(i), upconv)
            # output part

            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord, concat, max_len)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i < layers-1 and use_attn:
                attn = Attn_Net(ngf * mult, None, use_spect)
                # attn = Auto_Attn(ngf*mult, None, conv_dilation=conv_dilation)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None, actionmap=None, res_base=None,if_pos_enc=True):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """
        results = []
        f = self.generator(z, actionmap,if_pos_enc)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i))
             f = generator(f)

        # the features come from mask regions and valid regions, we directly add them together
        out = f + f_m
        output = self.outconv_g(out)
        # if res_base is not None:
        #     output = output + res_base
        results.append(output)
        if self.use_attn:
            out1 = self.attn_g(out, f_e[-1])
            out = torch.cat([out1, output], dim=1)
        else:
            out = torch.cat([out, output], dim=1)

        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out, actionmap,if_pos_enc)
            use_attn = (i < self.layers-1 and self.use_attn)
            if use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out1 = model(out, f_e[-2-i])
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                # if res_base is not None:
                #     output = output + res_base
                if i == self.layers-1:
                    output = self.mean_filter(output)
                results.append(output)
                if use_attn:
                    out = torch.cat([out1, output], dim=1)
                else:
                    out = torch.cat([out, output], dim=1)

        return results, attn

class ResGenerator(nn.Module):
    """
    ResNet Generator Network
    :param output_nc: number of channels in output
    :param ngf: base filter channel
    :param z_nc: latent channels
    :param img_f: the largest feature channels
    :param L: Number of refinements of density
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param output_scale: Different output scales
    """
    def __init__(self, output_nc=3, ngf=64, z_nc=128, img_f=1024, L=1, layers=6, norm='batch', activation='ReLU',
                 output_scale=1, use_spect=True, use_coord=False, use_attn=True,concat=False,frame_num=5000, use_spade=False, action_dim=300, use_action=False, conv_dilation=False):
        super(ResGenerator, self).__init__()

        self.layers = layers
        self.L = L
        self.output_scale = output_scale
        self.use_attn = use_attn
        self.use_action = use_action
        max_len = frame_num

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        # latent z to feature
        mult = min(2 ** (layers-1), img_f // ngf)
        self.generator = ResBlock(z_nc, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation)

        # set mean convoution
        kernel_size = 5
        self.mean_filter = nn.Sequential(nn.ReflectionPad1d(int(kernel_size / 2)),
                                               nn.AvgPool1d(kernel_size=kernel_size, stride=1))

        # transform
        for i in range(self.L):
            block = ResBlock(ngf * mult, ngf * mult, ngf * mult, None, nonlinearity, 'none', use_spect, use_coord, concat, max_len, conv_dilation)
            setattr(self, 'generator' + str(i), block)

        # decoder part
        for i in range(layers):

            mult_prev = mult
            mult = max(min(2 ** (layers - i - 2), img_f // ngf),2**0)
            if i > layers - output_scale:
                
                upconv = ResBlockDecoder(ngf * mult_prev + output_nc, ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_spade, action_dim, max_len, use_action)
            else:
               
                upconv = ResBlockDecoder(ngf * mult_prev , ngf * mult, ngf * mult, norm_layer, nonlinearity, use_spect, use_spade, action_dim, max_len, use_action)
            max_len = max_len * 2
            setattr(self, 'decoder' + str(i), upconv)
            # output part
            self.reduce_layer = reduce_layer
            if reduce_layer:
                self.attn_layer = 0
            else:
                self.attn_layer = 1
            if i > layers - output_scale - 1:
                outconv = Output(ngf * mult, output_nc, 3, None, nonlinearity, use_spect, use_coord, concat, max_len)
                setattr(self, 'out' + str(i), outconv)
            # short+long term attention part
            if i == self.attn_layer and use_attn:
                attn = Auto_Attn(ngf*mult, None, conv_dilation=conv_dilation)
                setattr(self, 'attn' + str(i), attn)

    def forward(self, z, f_m=None, f_e=None, mask=None, actionmap=None):
        """
        ResNet Generator Network
        :param z: latent vector
        :param f_m: feature of valid regions for conditional VAG-GAN
        :param f_e: previous encoder feature for short+long term attention layer
        :return results: different scale generation outputs
        """

        f = self.generator(z)
        for i in range(self.L):
             generator = getattr(self, 'generator' + str(i))
             f = generator(f)

        # the features come from mask regions and valid regions, we directly add them together
        out = f_m + f
        results= []
        attn = 0
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out, actionmap)
            if i == self.attn_layer and self.use_attn:
                # auto attention
                model = getattr(self, 'attn' + str(i))
                out, attn = model(out, f_e, mask)
            if i > self.layers - self.output_scale - 1:
                model = getattr(self, 'out' + str(i))
                output = model(out)
                if i == self.layers-1:
                    output = self.mean_filter(output)
                results.append(output)
                out = torch.cat([out, output], dim=1)

        return results, attn


class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True, concat=False, frame_num=5000, action_dim=300, use_action=False, conv_dilation=False, use_one_hot=False, num_classes=15):
        super(ResDiscriminator, self).__init__()

        self.layers = layers
        self.use_attn = use_attn
        self.use_action = use_action
        self.use_one_hot = use_one_hot

        if self.use_action:
            if use_one_hot:
                input_nc += num_classes
            else:
                self.conv_action = nn.Conv1d(action_dim, input_nc, 1)
                input_nc *= 2

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity
        max_len = frame_num
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf, norm_layer, nonlinearity, use_spect, use_coord, concat,max_len, conv_dilation)
        max_len = max_len//2
        mult = 1

        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 1 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer, conv_dilation)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord, concat,max_len, conv_dilation)
            setattr(self, 'encoder' + str(i), block)
            max_len = max_len // 2

        self.block1 = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord=False, concat=False, max_len=max_len, conv_dilation=conv_dilation)
        self.conv = SpectralNorm(nn.Conv1d(ndf * mult, 1, 3))

    def forward(self, x, actionmap=None):
        if actionmap is not None and self.use_action:
            actionmap = F.interpolate(actionmap, size=x.size()[2:], mode='nearest')
            if self.use_one_hot:
                x = torch.cat((x, actionmap), 1)
            else:
                x = torch.cat((x, self.conv_action(actionmap)), 1)
        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 1 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.block1(out)
        out = self.conv(self.nonlinearity(out))
        return out

class ResDiscriminator_classifier(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True, concat=False, frame_num=5000, action_dim=300, use_action=False, conv_dilation=False,num_classes=15):
        super(ResDiscriminator_classifier, self).__init__()

        self.layers = layers
        self.use_attn = use_attn
        self.use_action = use_action

        # if self.use_action:
        #     self.conv_action = nn.Conv1d(action_dim, input_nc, 1)
        #     input_nc *= 2

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity
        max_len = frame_num
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord, concat,max_len, conv_dilation)
        max_len = max_len//2
        mult = 1

        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer, conv_dilation)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord, concat,max_len, conv_dilation)
            setattr(self, 'encoder' + str(i), block)
            max_len = max_len // 2

        self.block1_dis = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord=False, concat=False, max_len=max_len, conv_dilation=conv_dilation)
        self.conv_dis = SpectralNorm(nn.Conv1d(ndf * mult, 1, 3))

        self.block1_class = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord=False, concat=False, max_len=max_len, conv_dilation=conv_dilation)
        self.conv_class = nn.Conv1d(ndf * mult, ndf * mult, 3)
        self.conv_final = nn.Conv1d(ndf * mult, num_classes, 1)


    def forward(self, x):

        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        # discriminator branch
        score = self.block1_dis(out)
        score = self.conv_dis(self.nonlinearity(score))
        # classifier
        class_pred = self.conv_class(self.nonlinearity(self.block1_class(out)))
        class_pred = self.conv_final(class_pred)

        return score, class_pred

class ResClassifier(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False, use_attn=True, concat=False, frame_num=5000, conv_dilation=False,num_classes=15):
        super(ResClassifier, self).__init__()

        self.layers = layers
        self.use_attn = use_attn


        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity
        max_len = frame_num
        # encoder part
        self.block0 = ResBlockEncoderOptimized(input_nc, ndf,norm_layer, nonlinearity, use_spect, use_coord, concat,max_len, conv_dilation)
        max_len = max_len//2
        mult = 1

        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f // ndf)
            # self-attention
            if i == 2 and use_attn:
                attn = Auto_Attn(ndf * mult_prev, norm_layer, conv_dilation)
                setattr(self, 'attn' + str(i), attn)
            block = ResBlock(ndf * mult_prev, ndf * mult, ndf * mult_prev, norm_layer, nonlinearity, 'down', use_spect, use_coord, concat,max_len, conv_dilation)
            setattr(self, 'encoder' + str(i), block)
            max_len = max_len // 2



        self.block1_class = ResBlock(ndf * mult, ndf * mult, ndf * mult, norm_layer, nonlinearity, 'none', use_spect, use_coord=False, concat=False, max_len=max_len, conv_dilation=conv_dilation)
        self.conv_class = nn.Conv1d(ndf * mult, ndf * mult, 3)
        self.conv_final = nn.Conv1d(ndf * mult, num_classes, 1)


    def forward(self, x, class_one_pred=False):

        out = self.block0(x)
        for i in range(self.layers - 1):
            if i == 2 and self.use_attn:
                attn = getattr(self, 'attn' + str(i))
                out, attention = attn(out)
            model = getattr(self, 'encoder' + str(i))
            out = model(out)

        # classifier
        class_pred = self.conv_class(self.nonlinearity(self.block1_class(out)))
        if class_one_pred:
            class_pred = F.avg_pool1d(class_pred, class_pred.size()[-1])
        class_pred = self.conv_final(class_pred)
        return class_pred

