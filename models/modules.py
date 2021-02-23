import jittor as jt 
from jittor.misc import _pair
from jittor import nn,init 
import math 

def cal_same_padding(x,kernel_size,stride,dilation):
    N,C,H,W = x.shape
    KH,KW = _pair(kernel_size)
    stride = _pair(stride)
    dilation = _pair(dilation)
    oH = (H+stride[0]-1)//stride[0]
    oW = (W+stride[1]-1)//stride[1]
    padding_h = max(0,(oH-1)*stride[0]+(KH-1)*dilation[0]+1-H)
    padding_w = max(0,(oW-1)*stride[1]+(KW-1)*dilation[1]+1-W)

    if padding_h%2!=0 or padding_w%2!=0:
        x = x.reindex([N,C,H+(padding_h%2),W+(padding_w%2)],["i0","i1","i2","i3"])
    return x,(padding_h//2,padding_w//2)


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, ceil_mode=False, count_include_pad=True):
        self.layer = nn.Pool(kernel_size=kernel_size, stride=stride,ceil_mode=ceil_mode, count_include_pad=count_include_pad, op="mean")
    
    def execute(self, x):
        x,padding = cal_same_padding(x,self.layer.kernel_size,self.layer.stride,(1,1))
        assert padding[0]==padding[1]
        padding = padding[0]
        self.layer.padding = padding
        return self.layer(x)


class SameConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = (0,0)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.is_depthwise_conv = self.groups == self.out_channels and self.groups == self.in_channels
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        Kh, Kw = self.kernel_size
        self.groups = groups
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'

        # self.weight = init.relu_invariant_gauss([out_channels, in_channels//groups, Kh, Kw], dtype="float", mode="fan_out")
        self.weight = init.invariant_uniform([out_channels, in_channels//groups, Kh, Kw], dtype="float")
        if bias:
            fan=1
            for i in self.weight.shape[1:]:
                fan *= i
            bound = 1 / math.sqrt(fan)
            self.bias = init.uniform([out_channels], dtype="float", low=-bound, high=bound)
        else:
            self.bias = None

    def execute(self, x):
        x,self.padding = cal_same_padding(x,self.kernel_size,self.stride,self.dilation)
        if self.groups == 1:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            assert C==self.in_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            assert oh>0 and ow>0
            xx = x.reindex([N,self.out_channels,C,oh,ow,Kh,Kw], [
                'i0', # Nid
                'i2', # Cid
                f'i3*{self.stride[0]}-{self.padding[0]}+i5*{self.dilation[0]}', # Hid+Khid
                f'i4*{self.stride[1]}-{self.padding[1]}+i6*{self.dilation[1]}', # Wid+KWid
            ])
            ww = self.weight.broadcast(xx.shape, [0,3,4])
            yy = xx*ww
            y = yy.sum([2,5,6]) # Kc, Kh, Kw
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y
        else:
            N,C,H,W = x.shape
            Kh, Kw = self.kernel_size
            G = self.groups
            CpG = C // G # channels per group
            assert C==self.in_channels
            oc = self.out_channels
            oh = (H+self.padding[0]*2-Kh*self.dilation[0]+self.dilation[0]-1)//self.stride[0]+1
            ow = (W+self.padding[1]*2-Kw*self.dilation[1]+self.dilation[1]-1)//self.stride[1]+1
            assert oh>0 and ow>0
            xx = x.reindex([N,G,oc//G,CpG,oh,ow,Kh,Kw], [
                'i0', # Nid
                f'i1*{CpG}+i3', # Gid
                f'i4*{self.stride[0]}-{self.padding[0]}+i6*{self.dilation[0]}', # Hid+Khid
                f'i5*{self.stride[1]}-{self.padding[1]}+i7*{self.dilation[1]}', # Wid+KWid
            ])
            # w: [oc, CpG, Kh, Kw]
            ww = self.weight.reindex([N, G, oc//G, CpG, oh, ow, Kh, Kw], [
                f'i1*{oc//G}+i2',
                'i3',
                'i6',
                'i7'
            ])
            ww.compile_options = xx.compile_options = {"G":G,"C":C}
            yy = xx*ww
            y = yy.reindex_reduce('add', [N, oc, oh, ow], [
                'i0',
                f'i1*{oc//G}+i2',
                'i4',
                'i5'
            ])
            if self.bias is not None:
                b = self.bias.broadcast(y.shape, [0,2,3])
                y = y + b
            return y          