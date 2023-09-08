import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'max_pool_3x1' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'max_pool_5x1' : lambda C, stride, affine: nn.MaxPool2d(5, stride=stride, padding=2),
  'avg_pool_3x1' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1),
  'avg_pool_5x1' : lambda C, stride, affine: nn.AvgPool2d(5, stride=stride, padding=2),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else ValueError('Stride must be 1 for skip_connect'),
  'conv_1x1' : lambda C, stride, affine: ReLUConvBN(C, C, (1,1), stride, (0,0), affine=affine),
  'conv_3x1' : lambda C, stride, affine: ReLUConvBN(C, C, (3,1), stride, (1,0), affine=affine),
  'conv_5x1' : lambda C, stride, affine: ReLUConvBN(C, C, (5,1), stride, (2,0), affine=affine),
  'conv_7x1' : lambda C, stride, affine: ReLUConvBN(C, C, (7,1), stride, (3,0), affine=affine),
  'conv_9x1' : lambda C, stride, affine: ReLUConvBN(C, C, (9,1), stride, (4,0), affine=affine),
  'diconv_3x1' : lambda C, stride, affine: ReLUDiConvBN(C, C, (3,1), stride, (2,0), (2,1), affine=affine),
  'diconv_5x1' : lambda C, stride, affine: ReLUDiConvBN(C, C, (5,1), stride, (4,0), (2,1), affine=affine),
}


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class ReLUDiConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(ReLUDiConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)