from .cnn_ig import RTCNN
from .cnn_w import RTWCNN
from .rnns import HARLSTM, HARBiLSTM, HARConvLSTM
from .t_fcn import FCNTSC
from .t_resnet import ResNetTSC
from .gtsnet import GTSNet

from .harblocks import *

def create_harmodel(arch, init_channels, NUM_CLASSES, window_size=None):
    # RTCNN
    if arch == 'RTCNN':
        # model = RTCNN(flat_size=None, init_channels=init_channels, num_classes=NUM_CLASSES, acc_num=None) # TODO: fix this
        raise NotImplementedError("%s is not included" % arch)
    elif arch == 'RTWCNN':
        model = RTWCNN(init_channels=init_channels, num_classes=NUM_CLASSES, segment_size=window_size)
    # LSTM
    elif arch == 'HARLSTM':
        model = HARLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'HARBiLSTM':
        model = HARBiLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'HARConvLSTM':
        model = HARConvLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    # TSC
    elif arch == 'ResNetTSC':
        model = ResNetTSC(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'FCNTSC':
        model = FCNTSC(init_channels=init_channels, num_classes=NUM_CLASSES)
    # GTSNet    
    elif arch == 'GTSNet':
        model = GTSNet(init_channels=init_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError("%s is not included" % arch)

    return model


def create_harblock(block_name, cnf):
    # LSTMBlock
    if block_name == 'LSTMBlock':
        block = LSTMBlock(cnf)
    # BiLSTMBlock
    elif block_name == 'BiLSTMBlock':
        block = BiLSTMBlock(cnf)
    # GTSResConvBlock
    elif block_name == 'GTSResConvBlock':
        block = GTSResConvBlock(cnf)
    else:
        raise ValueError("%s is not included" % block_name)
    
    return block