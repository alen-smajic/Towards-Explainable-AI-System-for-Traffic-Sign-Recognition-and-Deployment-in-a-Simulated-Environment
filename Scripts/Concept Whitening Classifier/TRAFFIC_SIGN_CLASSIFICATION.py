#from MODELS.model_resnet import *
import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torchvision import transforms
from torch.nn import init, Softmax
from iterative_normalization import IterNormRotation as cw_layer
from PIL import Image
import cv2



import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter

# import extension._bcnn as bcnn

__all__ = ['iterative_normalization', 'IterNorm']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(1.5, -0.5, g_tmp, P2)
            g_P.baddbmm_(1, -0.5, P2, g_tmp)
            g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1])
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                                                 self.eps, self.momentum, self.training)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


class IterNormRotation(torch.nn.Module):
    """
    Concept Whitening Module

    The Whitening part is adapted from IterNorm. The core of CW module is learning 
    an extra rotation matrix R that align target concepts with the output feature 
    maps.
    
    Because the concept activation is calculated based on a feature map, which
    is a matrix, there are multiple ways to calculate the activation, denoted
    by activation_mode.

    """
    def __init__(self, num_features, num_groups = 1, num_channels=None, T=10, dim=4, eps=1e-5, momentum=0.05, affine=False,
                mode = -1, activation_mode='pool_max', *args, **kwargs):
        super(IterNormRotation, self).__init__()
        assert dim == 4, 'IterNormRotation does not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.mode = mode
        self.activation_mode = activation_mode

        assert num_groups == 1, 'Please keep num_groups = 1. Current version does not support group whitening.'
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)

        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        #if self.affine:
        self.weight = Parameter(torch.Tensor(*shape))
        self.bias = Parameter(torch.Tensor(*shape))
        #else:
        #   self.register_parameter('weight', None)
        #   self.register_parameter('bias', None)

        #pooling and unpooling used in gradient computation
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool2d(kernel_size=3, stride=3)

        # running mean
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        # running rotation matrix
        self.register_buffer('running_rot', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        # sum Gradient, need to take average later
        self.register_buffer('sum_G', torch.zeros(num_groups, num_channels, num_channels))
        # counter, number of gradient for each concept
        self.register_buffer("counter", torch.ones(num_channels)*0.001)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def update_rotation_matrix(self):
        """
        Update the rotation matrix R using the accumulated gradient G.
        The update uses Cayley transform to make sure R is always orthonormal.
        """
        size_R = self.running_rot.size()
        with torch.no_grad():
            G = self.sum_G/self.counter.reshape(-1,1)
            R = self.running_rot.clone()
            for i in range(2):
                tau = 1000 # learning rate in Cayley transform
                alpha = 0
                beta = 100000000
                c1 = 1e-4
                c2 = 0.9
                
                A = torch.einsum('gin,gjn->gij', G, R) - torch.einsum('gin,gjn->gij', R, G) # GR^T - RG^T
                I = torch.eye(size_R[2]).expand(*size_R).cuda()
                dF_0 = -0.5 * (A ** 2).sum()
                # binary search for appropriate learning rate
                cnt = 0
                while True:
                    Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                    Y_tau = torch.bmm(Q, R)
                    F_X = (G[:,:,:] * R[:,:,:]).sum()
                    F_Y_tau = (G[:,:,:] * Y_tau[:,:,:]).sum()
                    dF_tau = -torch.bmm(torch.einsum('gni,gnj->gij', G, (I + 0.5 * tau * A).inverse()), torch.bmm(A,0.5*(R+Y_tau)))[0,:,:].trace()
                    if F_Y_tau > F_X + c1*tau*dF_0 + 1e-18:
                        beta = tau
                        tau = (beta+alpha)/2
                    elif dF_tau  + 1e-18 < c2*dF_0:
                        alpha = tau
                        tau = (beta+alpha)/2
                    else:
                        break
                    cnt += 1
                    if cnt > 500:
                        print("--------------------update fail------------------------")
                        print(F_Y_tau, F_X + c1*tau*dF_0)
                        print(dF_tau, c2*dF_0)
                        print("-------------------------------------------------------")
                        break
                print(tau, F_Y_tau)
                Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                R = torch.bmm(Q, R)
            
            self.running_rot = R
            self.counter = (torch.ones(size_R[-1]) * 0.001).cuda()


    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                                                 self.eps, self.momentum, self.training)
        # print(X_hat.shape, self.running_rot.shape)
        # nchw
        size_X = X_hat.size()
        size_R = self.running_rot.size()
        # ngchw
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])
        # updating the gradient matrix, using the concept dataset
        # the gradient is accumulated with momentum to stablize the training
        with torch.no_grad():
            # When 0<=mode, the jth column of gradient matrix is accumulated
            if self.mode>=0:
                if self.activation_mode=='mean':
                    self.sum_G[:,self.mode,:] = self.momentum * -X_hat.mean((0,3,4)) + (1. - self.momentum) * self.sum_G[:,self.mode,:]
                    self.counter[self.mode] += 1
                elif self.activation_mode=='max':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    max_values = torch.max(torch.max(X_test, 3, keepdim=True)[0], 4, keepdim=True)[0]
                    max_bool = max_values==X_test
                    grad = -((X_hat * max_bool.to(X_hat)).sum((3,4))/max_bool.to(X_hat).sum((3,4))).mean((0,))
                    self.sum_G[:,self.mode,:] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:,self.mode,:]
                    self.counter[self.mode] += 1
                elif self.activation_mode=='pos_mean':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    pos_bool = X_test > 0
                    grad = -((X_hat * pos_bool.to(X_hat)).sum((3,4))/(pos_bool.to(X_hat).sum((3,4))+0.0001)).mean((0,))
                    self.sum_G[:,self.mode,:] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:,self.mode,:]
                    self.counter[self.mode] += 1
                elif self.activation_mode=='pool_max':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    X_test_nchw = X_test.view(size_X)
                    maxpool_value, maxpool_indices = self.maxpool(X_test_nchw)
                    X_test_unpool = self.maxunpool(maxpool_value, maxpool_indices, output_size = size_X).view(size_X[0], size_R[0], size_R[2], *size_X[2:])
                    maxpool_bool = X_test == X_test_unpool
                    grad = -((X_hat * maxpool_bool.to(X_hat)).sum((3,4))/(maxpool_bool.to(X_hat).sum((3,4)))).mean((0,))
                    self.sum_G[:,self.mode,:] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:,self.mode,:]
                    self.counter[self.mode] += 1
            # # When mode > k, this is not included in the paper
            # elif self.mode>=0 and self.mode>=self.k:
            #     X_dot = torch.einsum('ngchw,gdc->ngdhw', X_hat, self.running_rot)
            #     X_dot = (X_dot == torch.max(X_dot, dim=2,keepdim=True)[0]).float().cuda()
            #     X_dot_unity = torch.clamp(torch.ceil(X_dot), 0.0, 1.0)
            #     X_G = torch.einsum('ngchw,ngdhw->gdchw', X_hat, X_dot_unity).mean((3,4))
            #     X_G[:,:self.k,:] = 0.0
            #     self.sum_G[:,:,:] += -X_G/size_X[0]
            #     self.counter[self.k:] += 1
        
        # We set mode = -1 when we don't need to update G. For example, when we train for main objective
        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
        X_hat = X_hat.view(*size_X)
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)

class ResidualNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch = 'resnet18', layers = [2,2,2,2], model_file = None):

        super(ResidualNetTransfer, self).__init__()
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file != None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            #print(checkpoint['best_prec1'])
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k,'bw','bn'): v for k,v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers

        """
        for whitened_layer in whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer-1].bn1 = cw_layer(64, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1 = cw_layer(128, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1 = cw_layer(256, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1 = cw_layer(512, activation_mode = 'pool_max')
        """

        for whitened_layer in whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer-1].bn1 = IterNormRotation(64, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer-layers[0]-1].bn1 = IterNormRotation(128, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer-layers[0]-layers[1]-1].bn1 = IterNormRotation(256, activation_mode = 'pool_max')
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer-layers[0]-layers[1]-layers[2]-1].bn1 = IterNormRotation(512, activation_mode = 'pool_max')

    def forward(self, x):
        return self.model(x)

    

folder = "rebalanced_data/"

transform = transforms.Resize((224,224), interpolation=Image.NEAREST)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResidualNetTransfer(44, 0, [5], arch = 'resnet18', layers = [2, 2, 2, 2]).to(device)

checkpoint = torch.load('checkpoints/blue_circle_red_triangle/ResNet18_traffic_signs44_CW_5_checkpoint.pth.tar', map_location='cpu')

model.load_state_dict(checkpoint['state_dict'])

model.eval()

#out_dict = {0:0, 1:1, 10:2, 11:3, 12:4, 13:5, 14:6, 15:7, 16:8, 17:9, 18:10, 19:11, 2:12, 20:13, 21:14, 22:15, 23:16, 24:17, 25:18, 26:19, 27:20, 28:21, 29:22, 3:23, 30:24, 31:25, 32:26, 33:27, 34:28, 35:29, 36:30, 37:31, 38:32, 39:33, 4:34, 40:35, 41:36, 42:37, 5:38, 6:39, 7:40, 8:41, 9:42}
out_dict = {0:0, 1:1, 2:10, 3:11, 4:12, 5:13, 6:14, 7:15, 8:16, 9:17, 10:18, 11:19, 12:2, 13:20, 14:21, 15:22, 16:23, 17:24, 18:25, 19:26, 20:27, 21:28, 22:29, 23:3, 24:30, 25:31, 26:32, 27:33, 28:34, 29:35, 30:36, 31:37, 32:38, 33:39, 34:4, 35:40, 36:41, 37:42, 38:42 , 39:5, 40:6, 41:7, 42:8, 43:9}

"""
for cat in os.listdir(folder):
    x = 0
    liste = os.listdir(folder + cat)
    for i in range(10):        
        path = random.choice(liste)
        
        img_path =  folder + cat + "/" + path
        print(img_path)

        

        #img = cv2.imread(img_path)

        img = Image.open(img_path)
        img_tensor = normalize(transforms.ToTensor()(transform(img)))
        out = model(img_tensor.unsqueeze(0).to(device))
        print("class", cat)
        #print(out.data[0].shape)
        #print(F.softmax(out.data[0]))
        print(out_dict[int(torch.argmax(F.softmax(out.data[0])))])
        #print(torch.argmax(out))
        #break
"""

transform = transforms.Resize((224,224), interpolation=Image.NEAREST)

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

img_path =  "rebalanced_data/13/00013_00004_00029.png"

img = cv2.imread(img_path)

img = Image.open(img_path)
img_tensor = normalize(transforms.ToTensor()(transform(img)))
#img_tensor = transform(transforms.ToTensor() (img))

out = model(img_tensor.unsqueeze(0).to(device))           
#print(out)
print(out_dict[int(torch.argmax(out))])

torch.onnx.export(model,                              # model being run
                  img_tensor.unsqueeze(0).to(device).float(), # model dummy input (or a tuple for multiple inputs)
                  "TSR_classificaton_model.onnx",     # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=12,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names = ['x'],                # the model's input names
                  output_names = ['y']                # the model's output names
                  )
