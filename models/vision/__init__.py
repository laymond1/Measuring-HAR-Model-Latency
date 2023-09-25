from .efficientnet import *
from .mnasnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .marnasnet import *


def create_vismodel(arch, init_channels, NUM_CLASSES):
        # mobilenet v2 & v3
    if arch == 'mobilenet_v2':
        model = mobilenet_v2(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'mobilenet_v3_small':
        model = mobilenet_v3_small(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'mobilenet_v3_large':
        model = mobilenet_v3_large(init_channels=init_channels, num_classes=NUM_CLASSES)
    # mnasnet
    elif arch == 'mnasnet0_5':
        model = mnasnet0_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'mnasnet0_75':
        model = mnasnet0_75(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'mnasnet1_0':
        model = mnasnet1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'mnasnet1_3':
        model = mnasnet1_3(init_channels=init_channels, num_classes=NUM_CLASSES)
    # shufflenet v2
    elif arch == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'shufflenet_v2_x1_5':
        model = shufflenet_v2_x1_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'shufflenet_v2_x2_0':
        model = shufflenet_v2_x2_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    # resnet & resnext
    elif arch == 'resnet18':
        model = resnet18(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnet34':
        model = resnet34(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnet50':
        model = resnet50(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnet101':
        model = resnet101(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnext50_32x4d':
        model = resnext50_32x4d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnext101_32x8d':
        model = resnext101_32x8d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'resnext101_64x4d':
        model = resnext101_64x4d(init_channels=init_channels, num_classes=NUM_CLASSES)
    # squeezenet
    elif arch == 'squeezenet1_0':    
        model = squeezenet1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'squeezenet1_1':    
        model = squeezenet1_1(init_channels=init_channels, num_classes=NUM_CLASSES)
    # efficientnet
    elif arch == 'efficientnet_b0':    
        model = efficientnet_b0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b1':    
        model = efficientnet_b1(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b2':    
        model = efficientnet_b2(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b3':    
        model = efficientnet_b3(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b4':    
        model = efficientnet_b4(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b5':    
        model = efficientnet_b5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b6':    
        model = efficientnet_b6(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_b7':    
        model = efficientnet_b7(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_v2_s':    
        model = efficientnet_v2_s(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_v2_m':    
        model = efficientnet_v2_m(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'efficientnet_v2_l':    
        model = efficientnet_v2_l(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'marnasnet_a':
        model = marnasnet_a(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'marnasnet_b':
        model = marnasnet_b(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'marnasnet_c':
        model = marnasnet_c(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'marnasnet_d':
        model = marnasnet_d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif arch == 'marnasnet_e':
        model = marnasnet_e(init_channels=init_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError("%s is not included" % arch)
    
    return model