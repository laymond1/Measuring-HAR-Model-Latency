import torch
import torch.nn as nn
    
    
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
    
    
class EAHARNAS(nn.Module):
    def __init__(self, input_nc, class_num):
        super(EAHARNAS, self).__init__()
        
        stem_multiplier = 3
        C_stem = stem_multiplier * input_nc
        n_ch = 48
        n_concat = 5
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_nc, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.conv1 = ReLUConvBN(C_stem, n_ch, 3, 1, 1) # conv_3x3
        self.conv2 = ReLUConvBN(C_stem, n_ch, 7, 1, 3) # conv_7x7
        self.conv3 = DilConv(C_stem, n_ch, 5, 1, 4, 2) # dil_conv_5x5
        self.conv4 = DilConv(C_stem, n_ch, 3, 1, 2, 2) # dil_conv_3x3
        self.conv5 = DilConv(C_stem, n_ch, 5, 1, 4, 2) # dil_conv_5x5
        self.conv6 = DilConv(C_stem, n_ch, 5, 1, 4, 2) # dil_conv_5x5
        
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1), stride=1, padding=(1,0))

        self.bilstm = nn.LSTM(n_ch*n_concat, 128, num_layers=2, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(128, class_num)

    
    def forward(self, x):
        # Conv Stem
        s0 = s1 = self.stem(x)
        # Cell
        s

        x = torch.transpose(x, 1, 2)
        x,(h, c) = self.lstm(x)

        h = h[-1,:,:]
        h = self.dropout(h)
        logits = self.fc(h)

        return logits
    

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)