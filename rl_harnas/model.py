import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_harnas.model_generator import BNConvReLU


class BNConvReLU(nn.Module):

    def __init__(self, input_channels, n_filters, kernel_size, padding):
        """ Generate a Conv-ReLU-BatchNorm block based on specified parameters.
            :input_channels: integer
                number of input channels
            :n_filters: integer
                number of convolutional filters to apply
            :kernel_size: integer
                size of convolutional kernel
            :padding: integer
                size of padding to append to each end of the input sequence"""

        super().__init__()

        self.conv = nn.Conv1d(input_channels, n_filters, kernel_size, padding=padding)
        self.ReLU = nn.ReLU()
        self.BN = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        """ Get output of block.
        x: tensor of shape (batch_size, input_channels, sequence_length), containing input data."""

        x = self.conv(x)
        x = self.ReLU(x)
        x = self.BN(x)

        return x


class RLNASModel(nn.Module):
    def __init__(self, C, num_classes):
        super(RLNASModel, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self.get_network()
        
        self.lstm = nn.LSTM(1184, 128, num_layers=2, batch_first=True) # 32 + 128 + 256 + 256*3
        self.classifier = nn.Linear(128, self._num_classes)

    def get_network(self):
        # make each layer            
        self.layer1, self.layer2 = BNConvReLU(self._C, 128, 1, 0), BNConvReLU(self._C, 32, 8, 3)
        self.layer3 = BNConvReLU(128, 256, 5, 2)
        self.layer4, self.layer5, self.layer6 =  \
            BNConvReLU(256, 256, 8, 3), BNConvReLU(256, 256, 8, 3), BNConvReLU(256, 256, 8, 3)
        self.layer8 = BNConvReLU(256, 256, 3, 1)   
        # make layers
        self._layers = nn.ModuleList([
            self.layer1, self.layer2, 
            self.layer3, 
            self.layer4, self.layer5, self.layer6, 
            self.layer8
        ])       
        
    def forward(self, x):
        # input x -> x1, x2
        x1, x2 = self.layer1(x), F.pad(self.layer2(x), pad=(0,1))
        # input x1 -> x3
        x3 = F.pad(self.layer3(x1), pad=(0,1))
        # input x3 -> x4, x5, x6
        x4, x5, x6 = self.layer4(x3), self.layer5(x3), self.layer6(x3)
        x7, x8 = torch.cat([x1, x4], dim=1), self.layer8(x4)    
        
        x = torch.cat([x2, x5, x6, x7, x8], dim=1)
        
        x = x.squeeze(-1).permute([0, 2, 1])
        lstm_out, (hn, cn) = self.lstm(x)
        last_time_step  = lstm_out[:, -1, :]
        logits = self.classifier(last_time_step)
        return logits
    
        
        