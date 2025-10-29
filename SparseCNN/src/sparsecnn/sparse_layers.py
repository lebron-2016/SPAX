import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_2_t
import torch.profiler.profiler as profiler
from torch.autograd.profiler import record_function
from collections import OrderedDict
from enum import Enum
import torch.nn.functional as F

from sparsecnn.cuda_kernels import sparse_concatenate, sparse_conv, sparse_deconv, \
    sparse_activation, sparsify, sparse_add_tensors, sparse_upsample, sparse_pooling, sparse_add_to_dense_tensor, sparse_mul_add
from sparsecnn.filter_conversion import convert_filter_out_channels_last, convert_half_filter

from .logging_layers import ComputationsLogger, InputLogger, PrevInputLogger, MultiplicationsLogger, OutputLogger

def convert_activation_string(name, dim_for_softmax=0, **kwargs):
    if name is None or name == "" or name.lower() == "linear":
        return None, -1
    if name.lower() == "relu":
        return torch.nn.ReLU(), 1
    elif name.lower() == "relu6":
        return torch.nn.ReLU6(), 2
    elif name.lower() == "leaky":
        return torch.nn.LeakyReLU(kwargs.get('leaky_relu_negative_slope', 1e-2)), 3
    elif name.lower() == "sigmoid":
        return torch.nn.Sigmoid(), 4
    elif name.lower() == "softmax":
        return torch.nn.Softmax(dim_for_softmax), 5
    elif name.lower() == "swish":
        return (lambda x: torch.sigmoid(x).mul_(x)), 6
    elif name.lower() == "tanh":
        return torch.nn.Tanh(), 7
    else:
        raise Exception(f"Activation {name} not implemented")

class SPBackend(Enum):
    cudnn = 0
    sparsecnn = 1
    sparse_cudnn = 2

    @classmethod
    def parse_string(cls, x):
        x = x.lower()
        if x == "cudnn":
            return SPBackend.cudnn
        if x == "sparsecnn":
            return SPBackend.sparsecnn
        if x == "sparse_cudnn":
            return SPBackend.sparse_cudnn
        raise Exception(f"invalid backend {x}")


class SPModule(nn.Module):
    mask_backup = None

    def __init__(self):
        super().__init__()

    def process_filters(self):
        modules = list(self.modules())

        for mod in modules:
            if type(mod) in [SPConv2d, SPConvTranspose2d]:
                mod.process_filters()


def to_tuple(*args) -> _size_2_t:
    result = []
    for x in args:
        if type(x) == int:
            result.append((x, x))
        elif len(x) == 1:
            result.append((x[0], x[0]))
        else:
            result.append(x)

    if len(result) == 1:
        return result[0]
    return result


class SPConv2d(nn.Conv2d, SPModule):
    backend = SPBackend.sparsecnn
    out_masks = []
    flops_sum = 0
    measure_flops = False

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            sparse_threshold: float = None,
            name: str = None,
            activation: str = None,
            dense_out=False,
            backend=None,
            use_logging=None,
            **kwargs
    ):
        kernel_size, stride, padding, dilation = to_tuple(kernel_size, stride, padding, dilation)

        super(SPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.name = name
        self.activation = None
        self.activation_int = 0
        self.backend = backend if backend is not None else SPConv2d.backend

        self.dense_in = False
        self.mask = None
        self.out_mask = None
        self.out_shape = []
        self.kwargs = kwargs
        self.in_shape = None
        assert kwargs.get('leaky_relu_negative_slope', 0.1) == 0.1, "Leaky ReLU implementation uses hard coded negative slope of 0.1"


    def __repr__(self):
        return f"MyConv2d{self.conv_idx} cin={self.in_channels} cout={self.out_channels}"


    def forward(self, input):

        return self._forward_sparse_conv(input)


    def _forward_sparse_conv(self, input):
        x = input
        mask = self.mask_backup[x.shape[2]].clone()

        out = sparse_conv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )

        if SPConv2d.measure_flops and self.groups == 1:
            mask = self.mask_backup[input.shape[2]].clone().float()
            out_mask = torch.conv2d(mask, torch.ones((1,1,self.weight.shape[1],self.weight.shape[2])).to(self.weight), None, self.stride, self.padding, self.dilation, self.groups)
            n_updated = (out_mask > 0).sum().int().item()
            SPConv2d.flops_sum += (n_updated * self.weight.numel())

        return out[0]

    def process_filters_half(self):
        self.orig_weights = self.weight.data.clone()
        result = super().half()
        if self.backend == SPBackend.sparsecnn:
            pixel_wise = self.groups == self.out_channels and self.groups == self.in_channels
            result.weight.data = convert_half_filter(result.weight.data, pixel_wise=pixel_wise)

        return result

    def process_filters_single(self):
        self.orig_weights = self.weight.data.clone()
        if self.backend == SPBackend.sparsecnn:
            self.weight.data = convert_filter_out_channels_last(self.weight.data)

        return self

    def process_filters(self):
        if self.weight.data.dtype == torch.float16:
            self.process_filters_half()
        else:
            self.process_filters_single()


class SPConvTranspose2d(nn.ConvTranspose2d, SPModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            output_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_2_t = 1,
            padding_mode: str = 'zeros',
            sparse_threshold: float = None,
            name: str = None,
            activation: str = None,
            store_prev_in=False,
            dense_out=False,
            backend=None,
            use_logging=None,
            **kwargs
    ):
        kernel_size, stride, padding, dilation = to_tuple(kernel_size, stride, padding, dilation)

        super(SPConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        
        self.name = name
        self.activation = None
        self.activation_int = 0
        self.backend = backend if backend is not None else SPConv2d.backend

        self.dense_in = False
        self.mask = None
        self.out_mask = None
        self.out_shape = []
        self.kwargs = kwargs
        self.in_shape = None
        assert kwargs.get('leaky_relu_negative_slope', 0.1) == 0.1, "Leaky ReLU implementation uses hard coded negative slope of 0.1"

    def __repr__(self):
        return f"MyConv2d{self.conv_idx} cin={self.in_channels} cout={self.out_channels}"


    def forward(self, input):

        assert(self.backend == SPBackend.cudnn or self.backend == SPBackend.sparsecnn)

        return self._forward_sparse_conv(input)

    def _forward_sparse_conv(self, input):
        x = input 
        mask = self.mask_backup[x.shape[2]].clone()

        out = sparse_deconv(
            x=x,
            filter=self.weight,
            mask=mask,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            c_out=self.out_channels,
            create_out_mask=True,
            out_mask=self.out_mask,
            out_shape=self.out_shape
        )

        if SPConv2d.measure_flops and self.groups == 1:
            mask = self.mask_backup[input.shape[2]].clone().float()
            out_mask = F.conv_transpose2d(mask, torch.ones((1,1,self.weight.shape[1],self.weight.shape[2])).to(self.weight), None, self.stride, self.padding, self.dilation, self.groups)
            n_updated = (out_mask > 0).sum().int().item()
            SPConv2d.flops_sum += (n_updated * self.weight.numel())

        return out[0]


    def process_filters_half(self):
        self.orig_weights = self.weight.data.clone()
        result = super().half()
        if self.backend == SPBackend.sparsecnn:
            pixel_wise = self.groups == self.out_channels and self.groups == self.in_channels
            result.weight.data = convert_half_filter(result.weight.data, pixel_wise=pixel_wise, transposed=True)

        return result

    def process_filters_single(self):
        self.orig_weights = self.weight.data.clone()
        if self.backend == SPBackend.sparsecnn:
            self.weight.data = convert_filter_out_channels_last(self.weight.data, transposed=True)

        return self

    def process_filters(self):
        if self.weight.data.dtype == torch.float16:
            self.process_filters_half()
        else:
            self.process_filters_single()


class SPSparsify(SPModule):

    def __init__(self, name="", sparse_threshold=None, dilation=3):
        super(SPSparsify, self).__init__()
        self.name = name
        self.dilation = dilation
        self.mask = None

    def forward(self, input, pred_mask): # mask 1表示要计算的
        if SPConv2d.backend != SPBackend.sparsecnn and SPConv2d.backend != SPBackend.sparse_cudnn:
            return input

        mask_backup = {}
        if self.dilation > 1:

            x = input.clone()
            
            mask_down2 = torch.max_pool2d(pred_mask.clone(), kernel_size=2, stride=2)
            mask_down4 = torch.max_pool2d(pred_mask.clone(), kernel_size=4, stride=4) 
            mask_down8 = torch.max_pool2d(pred_mask.clone(), kernel_size=8, stride=8)
            mask_down16 = torch.max_pool2d(pred_mask.clone(), kernel_size=16, stride=16)
            
            pred_mask = pred_mask > 0.

            pred_mask = pred_mask.int()
            mask_down2 = mask_down2.int()
            mask_down4 = mask_down4.int()
            mask_down8 = mask_down8.int()
            mask_down16 = mask_down16.int()

            mask_backup[input.shape[2]] = pred_mask
            mask_backup[input.shape[2] // 2] = mask_down2
            mask_backup[input.shape[2] // 4] = mask_down4
            mask_backup[input.shape[2] // 8] = mask_down8
            mask_backup[input.shape[2] // 16] = mask_down16

        return mask_backup


class SPBatchNorm2d(nn.BatchNorm2d, SPModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            inplace=True
    ):
        super(SPBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.first_iter = True
        self.scale = None
        self.offset = None
        self.inplace = inplace

    def convert_to_scale_offset(self, input):
        bn_scale = self.weight * torch.rsqrt(self.running_var + self.eps)

        self.scale = bn_scale
        self.offset = -self.running_mean * bn_scale + self.bias

        self.scale = self.scale[None, :, None, None]
        self.offset = self.offset[None, :, None, None]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        use_offset = True
        mask = None

        mask = self.mask_backup[input.shape[2]]

        if self.first_iter:
            self.first_iter = False
            use_offset = True
            self.convert_to_scale_offset(input)

        out = input
        out_mask = mask

        bias = None
        if use_offset:
            bias = self.offset
        if self.inplace:
            sparse_mul_add(out, out_mask, out, out_mask, self.scale, bias)
        else:
            out = torch.empty_like(input)
            out_mask = mask.clone()
            sparse_mul_add(input, mask, out, out_mask, self.scale, bias)


        return out


class SPActivation(SPModule):

    def __init__(self, name="", inplace=True, sparse_threshold=None, activation="relu", dim_for_softmax=0):
        super(SPActivation, self).__init__()
        self.name = name
        self.inplace = inplace
        self.activation, self.activation_int = convert_activation_string(activation, dim_for_softmax=dim_for_softmax)

    def forward(self, input):

        val, mask, mask_backup = input

        if not self.inplace:
            val = val.clone()
            mask = mask.clone()
    
        val = self.activation(val)
        return val, mask, mask_backup
