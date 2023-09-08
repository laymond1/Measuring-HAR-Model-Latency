import torch
import torch.nn as nn
from operations import *


class ConvCell(nn.Module):

  def __init__(self, genotype, C_prev, C):
    super(ConvCell, self).__init__()
    print(C_prev, C)

    self.stem0 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self.stem1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      op = OPS[name](C, 1, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, x):
    s0 = self.stem0(x)
    s1 = self.stem1(x)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkHAR(nn.Module):

  def __init__(self, C, num_classes, layers, genotype, classifier):
    super(NetworkHAR, self).__init__()
    self._layers = layers
    self._classifier = classifier

    multiplier = len(genotype.normal_concat) # 4
    stem_multiplier = 3
    # C_curr = stem_multiplier*C
    # self.stem = nn.Sequential(
    #   nn.Conv2d(C, C_curr, 3, padding=1, bias=False),
    #   nn.BatchNorm2d(C_curr)
    # )
    
    # no stem
    C_curr = 48
    self.cell = ConvCell(genotype, C, C_curr)
    C_prev = multiplier*C_curr
    
    # C_prev, C_curr = C_curr, multiplier*C_curr
    # self.cell = ConvCell(genotype, C_prev, C)
    # C_prev = multiplier*C
    
    # C_prev = C_curr
    # self.cell = ConvCell(genotype, C_prev, C_curr)
    # C_prev = multiplier*C_curr

    if self._classifier == 'LSTM':
      self.lstm = nn.LSTM(C_prev, 128, num_layers=2, batch_first=True)
      self.classifier = nn.Linear(128, num_classes)  
    else:
      self.global_pooling = nn.AdaptiveAvgPool2d(1)
      self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, x):
    # x = self.stem(x)
    x = self.cell(x)

    if self._classifier == "LSTM":
      x = x.squeeze(-1).permute([0, 2, 1])
      lstm_out, (hn, cn) = self.lstm(x)
      last_time_step  = lstm_out[:, -1, :]
      logits = self.classifier(last_time_step)
      return logits

    out = self.global_pooling(x)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits