import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.distributed as dist
import torch.nn as nn

class SyncMeanVar(Function):

    @staticmethod
    def forward(ctx, in_data, running_mean, running_var, momentum, training):
        if in_data.is_cuda:
            N, C, H, W = in_data.size()
            #print(in_data.size())
            in_data_t = in_data.transpose(0, 1).contiguous()
            #print(in_data_t.size())
            #print(C)
            in_data_t = in_data_t.view(C, -1)
            
            if training:
                mean_bn = in_data_t.mean(1)
                var_bn = in_data_t.var(1)

                sum_x = mean_bn ** 2 + var_bn
                dist.all_reduce(mean_bn)
                mean_bn /= dist.get_world_size()
                dist.all_reduce(sum_x)
                sum_x /= dist.get_world_size()
                var_bn = sum_x - mean_bn ** 2

                running_mean.mul_(momentum)
                running_mean.add_((1 - momentum) * mean_bn.data)
                running_var.mul_(momentum)
                running_var.add_((1 - momentum) * var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(running_mean)
                var_bn = torch.autograd.Variable(running_var)

            ctx.save_for_backward(in_data.data, mean_bn.data)
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return mean_bn, var_bn

    @staticmethod
    def backward(ctx, grad_mean_out, grad_var_out):
        if grad_mean_out.is_cuda:

            in_data, mean_bn = ctx.saved_tensors

            N, C, H, W = in_data.size()
            
            grad_mean_in = grad_mean_out.view(1,C,1,1)
            dist.all_reduce(grad_mean_in)
            grad_mean_in = grad_mean_in / H / N / W / dist.get_world_size()
            
            grad_var_in = grad_var_out.view(1,C,1,1)
            dist.all_reduce(grad_var_in)
            grad_var_in = 2 * (in_data - mean_bn.view(1,C,1,1)) / (H*N*W) * grad_var_in / dist.get_world_size()
            
            inDiff = grad_mean_in + grad_var_in
        else:
            raise RuntimeError('SyncBNFunc only support CUDA computation!')
        return inDiff, None, None, None, None

class SyncBatchNorm2d(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super(SyncBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        #self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        #self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        #self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        #self.register_buffer('running_var', torch.ones(1, num_features, 1))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.zero_()
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum}'
                .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, in_data):
        weight = self.weight.view(1, self.num_features, 1, 1)
        bias = self.bias.view(1, self.num_features, 1, 1)
        
        mean_bn, var_bn = SyncMeanVar.apply(
                    in_data, self.running_mean, self.running_var, self.momentum, self.training)
        
        N, C, H, W = in_data.size()
        
        mean_bn = mean_bn.view(1, C, 1)
        var_bn = var_bn.view(1, C, 1)
        
        in_data = in_data.view(N, C, -1)

        x_hat = (in_data - mean_bn) / (var_bn + self.eps).sqrt()
        x_hat = x_hat.view(N, C, H, W)
        out_data = x_hat * weight + bias
        return out_data