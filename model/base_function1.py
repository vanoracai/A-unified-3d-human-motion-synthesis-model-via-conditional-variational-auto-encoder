import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from .external_function import SpectralNorm
import math
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import copy

######################################################################################
# base function for network structure
######################################################################################


def init_weights(net, init_type='normal', gain=0.02, rank=-1):
    """Get different initial method for the network weights"""
    # print(rank)
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
            # if rank==0:
            #     a=1
            # if rank ==1:
            #     a=1
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    if rank <= 0:
        print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='batch', use_action=False):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer




def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.iter_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params/1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[], dist=False, rank=-1):
    """print the network structure and initial the network"""
    # print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        torch.backends.cudnn.benchmark = True
        net.cuda()

        if dist:
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()],find_unused_parameters=True)
        else:
            net = torch.nn.DataParallel(net, gpu_ids)

    init_weights(net, init_type, rank=rank)
    # if rank == 0:
    #     print(net.state_dict()['module.block0.model.1.module.bias'])
    # if rank == 1:
    #     print(net.state_dict()['module.block0.model.1.module.bias'])
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, concat=False, max_len=5000, **kwargs):
    """use coord convolution layer to add position information"""
    modules = []
    if 'padding' in kwargs and kwargs['padding']>0:
        modules.append(nn.ReplicationPad1d(kwargs['padding']))
        del kwargs['padding']
    if use_coord:
        modules.append(PosCov(input_nc, output_nc, concat, use_spect, max_len, **kwargs))
    else:
        modules.append(spectral_norm(nn.Conv1d(input_nc, output_nc, **kwargs), use_spect))
    return modules
        #return nn.Conv1d(input_nc, output_nc, **kwargs)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim-1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim-1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """
    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret

class PosCov(nn.Module):
    """
    Positional encoding into the convolution
    """
    def __init__(self, input_nc, output_nc, concat=False, use_spect=False, max_len=5000,**kwargs):
        super(PosCov, self).__init__()
        self.max_len = max_len
        self.concat = concat
        self.posencode = PositionalEncoding(d_model=input_nc, max_len=max_len)
        if concat:
            input_nc = input_nc *2
        self.conv = spectral_norm(nn.Conv1d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        # print(x.shape)
        # print(self.max_len)
        if self.concat:
            ret = torch.cat((x,self.posencode(x)), 1)
        else:
            ret = x + self.posencode(x)
        ret = self.conv(ret)

        return ret

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.2, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.max_len = max_len
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
        div_term = torch.exp(torch.arange(0, d_model, 2).type(torch.FloatTensor) * -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        if pe[:,1::2].shape[1] % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # 1 , d_model, max_len
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(2) <= self.max_len:
            return Variable(self.pe[:, :, :x.size(2)],requires_grad=False)
        elif x.size(2) > self.max_len:
            pad = (x.size(2)-self.max_len)//2
            return nn.ReflectionPad1d(pad)(Variable(self.pe[:, :, :x.size(2)],requires_grad=False))


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm1d, nonlinearity= nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False, concat=False, max_len=5000, conv_dilation=False, use_action=False, use_spade=False, action_dim=300, use_one_hot=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.sample = True
        self.use_action = use_action
        self.use_spade = use_spade
        self.norm_layer = norm_layer

        if sample_type == 'none':
            self.sample = False

        elif sample_type == 'down':
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        if conv_dilation:
            kwargs_1 = {'kernel_size': 3, 'stride': 1, 'padding': 4}
            kwargs_2 = {'kernel_size': 3, 'stride': 1, 'dilation' :3}
            kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        else:
            kwargs_1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
            kwargs_2 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
            kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1_module = coord_conv(input_nc, hidden_nc, use_spect, use_coord, concat, max_len, **kwargs_1)
        self.conv2_module = coord_conv(hidden_nc, output_nc, use_spect, use_coord, concat, max_len, **kwargs_2)
        self.bypass_module = coord_conv(input_nc, output_nc, use_spect, use_coord,concat, max_len, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, *self.conv1_module, nonlinearity, *self.conv2_module,)
            self.shortcut = nn.Sequential(*self.bypass_module, )
        else:
            if use_spade and use_action:
                self.spade1 = SPADE(norm_layer, norm_nc=input_nc, label_nc=action_dim, ks=3, max_len=max_len, use_one_hot=use_one_hot)
                self.spade2 = SPADE(norm_layer, norm_nc=hidden_nc, label_nc=action_dim, ks=3, max_len=max_len, use_one_hot=use_one_hot)
                self.model1 = nn.Sequential(nonlinearity, *self.conv1_module)
                self.model2 = nn.Sequential(nonlinearity, *self.conv2_module)
                self.spade_s = SPADE(norm_layer, norm_nc=input_nc, label_nc=action_dim, ks=3, max_len=max_len, use_one_hot=use_one_hot)
                self.shortcut = nn.Sequential(nonlinearity, *self.bypass_module)
            else:
                self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, *self.conv1_module, norm_layer(hidden_nc), nonlinearity, *self.conv2_module,)
                self.shortcut = nn.Sequential(norm_layer(input_nc), nonlinearity, *self.bypass_module, )


    def forward(self, x, actionmap=None,if_pos_enc=True):
        if actionmap is not None and self.use_action and self.use_spade and type(self.norm_layer) != type(None):
            forward_pass = self.model2(self.spade2(self.model1(self.spade1(x, actionmap,if_pos_enc)), actionmap,if_pos_enc))
            # forward_pass = self.model2(self.model1(self.spade1(x, actionmap)))
            bypass = self.shortcut(self.spade_s(x, actionmap,if_pos_enc))
        else:
            forward_pass = self.model(x)
            bypass = self.shortcut(x)

        if self.sample:
            out = self.pool(forward_pass) + self.pool(bypass)
        else:
            out = forward_pass + bypass

        return out


class ResBlockEncoderOptimized(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator and representation network
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm1d, nonlinearity= nn.LeakyReLU(), use_spect=False, use_coord=False, concat=False, max_len=5000, conv_dilation=False):
        super(ResBlockEncoderOptimized, self).__init__()
        self.conv_dilation = conv_dilation
        if conv_dilation:
            kwargs_1 = {'kernel_size': 3, 'stride': 1, 'padding': 4}
            kwargs_2 = {'kernel_size': 3, 'stride': 1, 'dilation' :3}
            kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        else:
            kwargs_1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
            kwargs_2 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
            kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1_module = coord_conv(input_nc, output_nc, use_spect, use_coord=False, **kwargs_1)
        self.conv2_module = coord_conv(output_nc, output_nc, use_spect, use_coord, concat, max_len, **kwargs_2)
        self.bypass_module = coord_conv(input_nc, output_nc, use_spect, use_coord=False, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(*self.conv1_module, nonlinearity, *self.conv2_module, nn.AvgPool1d(kernel_size=2, stride=2))
        else:
            self.model = nn.Sequential(*self.conv1_module, norm_layer(output_nc), nonlinearity, *self.conv2_module, nn.AvgPool1d(kernel_size=2, stride=2))

        self.shortcut = nn.Sequential(nn.AvgPool1d(kernel_size=2, stride=2), *self.bypass_module)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_spade=False, action_dim=300, max_len=500, use_action=False,use_one_hot=False):
        super(ResBlockDecoder, self).__init__()

        self.use_action = use_action
        self.use_spade = use_spade

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        kwargs_1 = {'kernel_size': 3, 'stride':1, 'padding':1}
        self.conv1_module = coord_conv(input_nc, output_nc, use_spect, **kwargs_1)
        self.conv2_module = [spectral_norm(nn.ConvTranspose1d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)]
        self.bypass_module = [spectral_norm(nn.ConvTranspose1d(input_nc, output_nc, kernel_size=3, stride=2,  padding=1, output_padding=1), use_spect)]

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, *self.conv1_module, nonlinearity, *self.conv2_module,)
        else:
            if use_spade and use_action:
                self.spade1 = SPADE(norm_layer, norm_nc=input_nc, label_nc=action_dim, ks=3, max_len=max_len,use_one_hot=use_one_hot)
                self.spade2 = SPADE(norm_layer, norm_nc=hidden_nc, label_nc=action_dim, ks=3, max_len=max_len,use_one_hot=use_one_hot)
                self.model1 = nn.Sequential(nonlinearity, *self.conv1_module)
                self.model2 = nn.Sequential(nonlinearity, *self.conv2_module)
            else:
                self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, *self.conv1_module, norm_layer(hidden_nc), nonlinearity, *self.conv2_module,)

        if use_spade and use_action:
            self.spade_s = SPADE(norm_layer, norm_nc=input_nc, label_nc=action_dim, ks=3, max_len=max_len,use_one_hot=use_one_hot)
            self.shortcut = nn.Sequential(nonlinearity, *self.bypass_module)
        else:
            self.shortcut = nn.Sequential(*self.bypass_module)

    def forward(self, x, actionmap = None, if_pos_enc=True):
        if actionmap is not None and self.use_action and self.use_spade:
            out = self.model2(self.spade2(self.model1(self.spade1(x, actionmap,if_pos_enc)),actionmap,if_pos_enc)) + self.shortcut(self.spade_s(x, actionmap,if_pos_enc))
            # out = self.model2(self.model1(self.spade1(x, actionmap))) + self.shortcut(self.spade_s(x, actionmap))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm1d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False, concat=False, max_len=5000):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1_module = coord_conv(input_nc, output_nc, use_spect, use_coord, concat, max_len, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad1d(int(kernel_size/2)), *self.conv1_module, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad1d(int(kernel_size / 2)), *self.conv1_module, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, input_size, output_size, dropout=0.1, use_spect=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert output_size % num_head == 0
        # We assume d_v always equals d_k
        self.d_k = output_size // num_head
        self.h = num_head
        self.linears = clones(spectral_norm(nn.Linear(input_size, output_size),use_spect),3)
        self.linear_final = spectral_norm(nn.Linear(output_size, output_size),use_spect)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None): #query, key, value, mask=None):
        "Implements Figure 2"
        n,_,t = query.size()
        query = query.permute(0,2,1).contiguous().view(n, t,-1)#n,t,c
        key = key.permute(0, 2, 1).contiguous().view(n, t, -1)  # n,t,c
        value = value.permute(0, 2, 1).contiguous().view(n, t, -1)  # n,t,c

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query1, key1, value1 = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query1, key1, value1, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x = self.linear_final(x)
        # change back to original
        x = x.view(n,t,-1).permute(0,2,1) #n,c,t

        return x

class Attn_Net(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, use_spect=False):
        super(Attn_Net, self).__init__()
        self.input_nc = input_nc
        self.self_attn = MultiHeadedAttention(num_head=4, input_size=input_nc, output_size=input_nc, use_spect=use_spect)
        self.context_attn = MultiHeadedAttention(num_head=4, input_size=input_nc, output_size=input_nc, use_spect=use_spect)
        self.norm_layer = norm_layer
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))
        if norm_layer is not None:
            self.norm1 = norm_layer(input_nc)
            self.norm2 = norm_layer(input_nc)

    def forward(self, x, pre=None):
        """
        inputs :
            x : input feature maps( B X C X T)
            pre: ( B X C X T)
        returns :
            out :  value ( B X C X T)
        """
        x = self.gamma*self.self_attn(x, x, x) + x
        if self.norm_layer is not None:
            x = self.norm1(x)

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            x = self.alpha*self.context_attn(x, pre, pre) + x
            if self.norm_layer is not None:
                x = self.norm2(x)

        return x

class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, conv_dilation=False):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv1d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc*1.5), input_nc, input_nc, norm_layer=norm_layer, use_spect=True, conv_dilation=conv_dilation)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X T)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        B, C, T = x.size()
        proj_query = self.query_conv(x).view(B, -1, T)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, T)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, T)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, T), attention.permute(0, 2, 1)).view(B, -1, T)
            context_flow = self.alpha * (1-mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention

class SPADE(nn.Module):
    def __init__(self, norm_layer, norm_nc, label_nc, ks=3, max_len=500,use_one_hot=False):#, nhidden=512):#config_text,
        super().__init__()

        self.param_free_norm = norm_layer(norm_nc, affine=False)
        self.posencode = PositionalEncoding(d_model=label_nc, max_len=max_len)
        self.use_one_hot = use_one_hot



        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv1d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        if use_one_hot:
            self.mlp_gamma = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv1d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        else:
            self.mlp_gamma = nn.Conv1d(label_nc, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv1d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, actionmap,if_pos_enc=True):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map, add positonal encoding
        actionmap = F.interpolate(actionmap, size=x.size()[2:], mode='nearest')
        if if_pos_enc:
            actv = actionmap + self.posencode(actionmap)
        else:
            actv = actionmap

        if self.use_one_hot: # use one hot label
            actv = self.mlp_shared(actv)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        if normalized.shape[0] == 2*(actionmap.shape[0]):
            out = normalized * (1 + gamma.repeat(2,1,1)) + beta.repeat(2,1,1)
        else:
            out = normalized * (1 + gamma) + beta

        return out

