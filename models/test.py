import torch

from vision import *

if __name__ == '__main__':
    input = torch.randn(1, 6, 128)
    # model = mobilenet_v2(init_channels=6, num_classes=6)
    # model = resnet18(init_channels=6, num_classes=6)
    # model = resnext50_32x4d(init_channels=6, num_classes=6)
    # model = mobilenet_v3_large(init_channels=6, num_classes=6)
    # model = mnasnet0_5(init_channels=6, num_classes=6)
    # model = shufflenet_v2_x1_0(init_channels=6, num_classes=6)
    # model = efficientnet_v2_s(init_channels=6, num_classes=6)
    model = squeezenet1_0(init_channels=6, num_classes=6)
    
    print(model(input).shape)