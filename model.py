import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
import ctlib
import numpy as np

import graph_laplacian_2

class adj_weight(Function):
    def __init__(self, k=9):
        self.k = k
    def forward(self, x):
        return graph_laplacian_2.forward(x, self.k)
    
class gcn_module(nn.Module):
    def __init__(self):
        super(gcn_module, self).__init__()
        
    def forward(self, x, adj):
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            ttt = torch.mm(adj[i], torch.transpose(x[i], 0, 1))
            out[i] = torch.transpose(ttt, 0, 1)
        return out
    
class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, weight, proj, options):
        temp = ctlib.projection(input_data, options, 0) - proj
        intervening_res = ctlib.backprojection(temp, options, 0)
        self.save_for_backward(intervening_res, weight, options)
        out = input_data - weight * intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, weight, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options, 0)
        temp = ctlib.backprojection(temp, options, 0)
        grad_input = grad_output - weight * temp
        temp = intervening_res * grad_output
        grad_weight = - temp.sum().view(-1)
        return grad_input, grad_weight, None, None
    
    
class projection(Function):
    @staticmethod
    def forward(self, input_data, options):
        temp = ctlib.projection(input_data, options, 0)
        intervening_res = ctlib.backprojection(temp, options, 0)
        self.save_for_backward(intervening_res, weight, options)
        out = input_data - weight * intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, weight, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options, 0)
        temp = ctlib.backprojection(temp, options, 0)
        grad_input = grad_output - weight * temp
        temp = intervening_res * grad_output
        grad_weight = - temp.sum().view(-1)
        return grad_input, grad_weight, None, None

class sigma_activation(nn.Module):
    def __init__(self, ddelta):
        super(sigma_activation, self).__init__()
        self.relu = nn.ReLU(inplace=True)      
        self.ddelta = ddelta
        self.coeff = 1.0 / (4.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu = self.relu(x_i)
        x_square = torch.mul(x_i, x_i)
        x_square *= self.coeff
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu, x_square + 0.5*x_i + 0.25 * self.ddelta)
    
class sigma_derivative(nn.Module):
    def __init__(self, ddelta):
        super(sigma_derivative, self).__init__()
        self.ddelta = ddelta
        self.coeff2 = 1.0 / (2.0 * self.ddelta)

    def forward(self, x_i):
        x_i_relu_deri = torch.where(x_i > 0, torch.ones_like(x_i), torch.zeros_like(x_i))
        return torch.where(torch.abs(x_i) > self.ddelta, x_i_relu_deri, self.coeff2 *x_i + 0.5)

class LDA(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(LDA, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
        
        ratio = 1024 / 64
        options_spase_view = torch.Tensor([views/ratio, dets, width, height, dImg, dDet, dAng * ratio, s2r, d2r, binshift])
        
        self.options = nn.Parameter(options, requires_grad=False)
        
        self.options_spase_view = nn.Parameter(options_spase_view, requires_grad=False)
        self.thresh = nn.Parameter(torch.Tensor([0.002]), requires_grad=True)
        
        self.sigma = 10**6
        
        channel_num = 48
        self.channel_num = channel_num
        
        self.conv0 = nn.Conv2d(1, channel_num, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.deconv0 = nn.ConvTranspose2d(channel_num, 1, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.conv0_s = nn.Conv2d(1, channel_num, kernel_size=3, padding=1)
        self.conv1_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv2_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.conv3_s = nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.deconv0_s = nn.ConvTranspose2d(channel_num, 1, kernel_size=3, padding=1)
        self.deconv1_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv2_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        self.deconv3_s = nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True) 
        self.act = sigma_activation(0.001)
        self.deri_act = sigma_derivative(0.001)
        self.delta_ = 10**-5
        
        self.block1 = gcn_module()
        
        self.adj_weight = adj_weight()
        
        self.c = 10**5
        
        self.alpha_list = []
        for block_index in range(block_num):
            eval_func1 = "self.alpha_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.alpha_list.append(" + "self.alpha_" + str(block_index) + ")"
            exec(eval_func2)
            
        self.beta_list = []
        for block_index in range(block_num):
            eval_func1 = "self.beta_" + str(block_index) + " = nn.Parameter(torch.Tensor([0.02]), requires_grad=True)"
            exec(eval_func1)
            eval_func2 = "self.beta_list.append(" + "self.beta_" + str(block_index) + ")"
            exec(eval_func2)
        
        
    def submodule_grad_R_exact(self, lambda_, x_x4_forward):
        
        [x1, x2, x3, x4] = x_x4_forward
        thresh = lambda_ * torch.abs(self.thresh)
        norm_along_d = torch.norm(x4, dim=1, keepdim=True)
        x_denominator = torch.where(norm_along_d > thresh, norm_along_d, thresh)
        x_denominator = x_denominator.repeat([1, self.channel_num, 1, 1])
        x4_out = torch.div(x4, x_denominator)

        x4_dec = torch.nn.functional.conv_transpose2d(x4_out, weight=self.conv3.weight, bias=None, stride=1, padding=1)
        x3_deri_act = self.deri_act(x3)
        x3_dec = torch.nn.functional.conv_transpose2d(torch.mul(x3_deri_act, x4_dec), weight=self.conv2.weight, bias=None, stride=1, padding=1)
        x2_deri_act = self.deri_act(x2)
        x2_dec = torch.nn.functional.conv_transpose2d(torch.mul(x2_deri_act, x3_dec), weight=self.conv1.weight, bias=None, stride=1, padding=1)
        x1_deri_act = self.deri_act(x1)
        x1_dec = torch.nn.functional.conv_transpose2d(torch.mul(x1_deri_act, x2_dec), weight=self.conv0.weight, bias=None, stride=1, padding=1)
        
        return x1_dec
    
    def submodule_grad_R(self, lambda_, x_x4_forward):
        
        [x1, x2, x3, x4] = x_x4_forward
        
        thresh = lambda_ * torch.abs(self.thresh)
        norm_along_d = torch.norm(x4, dim=1, keepdim=True)
        x_denominator = torch.where(norm_along_d > thresh, norm_along_d, thresh)
        x_denominator = x_denominator.repeat([1, self.channel_num, 1, 1])
        x4_out = torch.div(x4, x_denominator)
        
        x4_dec = self.deconv3(x4_out)
        x3_deri_act = self.deri_act(x3)
        x3_dec = self.deconv2(torch.mul(x3_deri_act, x4_dec))
        x2_deri_act = self.deri_act(x2)
        x2_dec = self.deconv1(torch.mul(x2_deri_act, x3_dec))
        x1_deri_act = self.deri_act(x1)
        x1_dec = self.deconv0(torch.mul(x1_deri_act, x2_dec))
        
        return x1_dec
    
    def submodule_grad_sinogram(self, x_x4_forward, adj):
        
        [x1, x2, x3, x4] = x_x4_forward
        
        # graph Laplacian
        x5 = nn.functional.unfold(x4, 8, dilation=1, padding=0, stride=8)
        x6 = self.block1(x5, adj)
        x6 = nn.functional.fold(x6, (1024, 512), 8, dilation=1, padding=0, stride=8)  # 1024 * 512 * channel   # 256 * 256
        
        x4_dec = self.deconv3_s(x6)
        x3_deri_act = self.deri_act(x3)
        x3_dec = self.deconv2_s(torch.mul(x3_deri_act, x4_dec))
        x2_deri_act = self.deri_act(x2)
        x2_dec = self.deconv1_s(torch.mul(x2_deri_act, x3_dec))
        x1_deri_act = self.deri_act(x1)
        x1_dec = self.deconv0_s(torch.mul(x1_deri_act, x2_dec))
        
        x1_dec = ctlib.backprojection(x1_dec, self.options, 0)
        
        return x1_dec
    
    def submodule_R(self, x_x4_forward):

        x4 = x_x4_forward[-1]    
        norm_out = torch.norm(x4, dim=1, keepdim=False)
        norm_out = torch.norm(norm_out, p=1, dim=[1, 2], keepdim=False)
        
        return norm_out
    
    def x4_forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(self.act(x1))
        x3 = self.conv2(self.act(x2))
        x4 = self.conv3(self.act(x3))
        return [x1, x2, x3, x4]

    def x4_forward_sinogram(self, x):
        x = ctlib.projection(x, self.options, 0)
        x1 = self.conv0_s(x)
        x2 = self.conv1_s(self.act(x1))
        x3 = self.conv2_s(self.act(x2))
        x4 = self.conv3_s(self.act(x3))
        return [x1, x2, x3, x4]
        
    def function_value(self, x, proj, x_x4_forward):
        
        ff_ = torch.norm(ctlib.projection(x, self.options, 0) - proj, p = 2, dim=[1, 2, 3], keepdim=False)
        f_ = 0.5 * ff_ * ff_ + self.submodule_R(x_x4_forward)
        return torch.mean(f_, dim=0, keepdim=False)
        
    def forward(self, input_data, proj, block_num):
        
        x = input_data
        lambda_ = nn.Parameter(torch.Tensor([1.0]), requires_grad=False).cuda()
        
        [x1, x2, x3, x4] = self.x4_forward_sinogram(x)
        
        
        # graph Laplacian
        patch1 = nn.functional.unfold(x4, 8, dilation=1, padding=0, stride=8)
        
        
        
        
        #patch1 = nn.functional.unfold(x, 10, dilation=1, padding=4, stride=2)
        
        
        patch1 = torch.transpose(patch1, 1, 2).contiguous()
        
        adj1 = []
        for i in range(input_data.size(0)):
            adj1.append(self.adj_weight.forward(patch1[i])) # [batch, 15876, 15876]
        
        for i in range(block_num):    
            # get the step size
            alpha = torch.abs(self.alpha_list[i])     
            beta = torch.abs(self.beta_list[i])
            
            #print(alpha)
            
            x_x4_forward = self.x4_forward(x)
            
            # compute b
            b = prj_fun.apply(x, alpha, proj, self.options_spase_view)
            
            grad_f_x = torch.div(x - b, alpha)
                
            # check if lambda_ should decrease
            if i >= 1:
                if_decrease_lambda = torch.mean(torch.norm(grad_f_x + self.submodule_grad_R(lambda_, x_x4_forward), p = 2, dim=[1, 2, 3], keepdim=False), dim=0, keepdim=False)
                lambda_ = torch.where(if_decrease_lambda < lambda_*torch.abs(self.thresh)*self.sigma, 0.9*lambda_, lambda_)
            
            # compute the candidate x_u and x_v
            input_second_regu = self.x4_forward_sinogram(b)
            x = b - beta * self.submodule_grad_R(lambda_, self.x4_forward(b)) - beta * self.submodule_grad_sinogram(input_second_regu, adj1)# change submodule_grad_R_exact to submodule_grad_R if using inexact tranpose
            
            
        return x
