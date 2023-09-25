import argparse
import os
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from utils import create_exp_dir

from data_providers import *
from models.vision import *


# if '1.6.0' not in torch.__version__:
#     raise ValueError("PyTorch version should be 1.6.0")


def convert_model(net, input_tensor):
    # convert model to torchscript
    traced_script_module = torch.jit.trace(net, input_tensor)
    model_path = os.path.join(args.save, f'{args.arch}.pt')
    torch.jit.save(traced_script_module, model_path)
    print(f"CPU model saved at : {model_path}")


def main(args):
    if len(args.save) < 1:
        args.save = 'mobile_pt/{}'.format(args.dataset)
    create_exp_dir(args.save, scripts_to_save=None)
    
    # dataset
    if args.dataset == 'uci':
        dataset = UCIHARDataProvider(data_path='', train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'opp':
        dataset = OPPDataProvider(data_path='', train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'kar':
        dataset = KUHARDataProvider(data_path='', train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'uni':
        dataset = UniMiBDataProvider(data_path='', train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'wis':
        dataset = WISDMDataProvider(data_path='', train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    else:
        raise ValueError("Unknown dataset type")
    
    init_channels, window_size = dataset.data_shape
    NUM_CLASSES = dataset.n_classes
    batch_size = args.batch_size
    
    # mobilenet v2 & v3
    if args.arch == 'mobilenet_v2':
        model = mobilenet_v2(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'mobilenet_v3_small':
        model = mobilenet_v3_small(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'mobilenet_v3_large':
        model = mobilenet_v3_large(init_channels=init_channels, num_classes=NUM_CLASSES)
    # mnasnet
    elif args.arch == 'mnasnet0_5':
        model = mnasnet0_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'mnasnet0_75':
        model = mnasnet0_75(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'mnasnet1_0':
        model = mnasnet1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'mnasnet1_3':
        model = mnasnet1_3(init_channels=init_channels, num_classes=NUM_CLASSES)
    # shufflenet v2
    elif args.arch == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'shufflenet_v2_x1_5':
        model = shufflenet_v2_x1_5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'shufflenet_v2_x2_0':
        model = shufflenet_v2_x2_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    # resnet & resnext
    elif args.arch == 'resnet18':
        model = resnet18(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnet34':
        model = resnet34(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnet50':
        model = resnet50(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnet101':
        model = resnet101(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnext50_32x4d':
        model = resnext50_32x4d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnext101_32x8d':
        model = resnext101_32x8d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'resnext101_64x4d':
        model = resnext101_64x4d(init_channels=init_channels, num_classes=NUM_CLASSES)
    # squeezenet
    elif args.arch == 'squeezenet1_0':    
        model = squeezenet1_0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'squeezenet1_1':    
        model = squeezenet1_1(init_channels=init_channels, num_classes=NUM_CLASSES)
    # efficientnet
    elif args.arch == 'efficientnet_b0':    
        model = efficientnet_b0(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b1':    
        model = efficientnet_b1(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b2':    
        model = efficientnet_b2(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b3':    
        model = efficientnet_b3(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b4':    
        model = efficientnet_b4(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b5':    
        model = efficientnet_b5(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b6':    
        model = efficientnet_b6(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_b7':    
        model = efficientnet_b7(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_v2_s':    
        model = efficientnet_v2_s(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_v2_m':    
        model = efficientnet_v2_m(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'efficientnet_v2_l':    
        model = efficientnet_v2_l(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'marnasnet_a':
        model = marnasnet_a(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'marnasnet_b':
        model = marnasnet_b(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'marnasnet_c':
        model = marnasnet_c(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'marnasnet_d':
        model = marnasnet_d(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'marnasnet_e':
        model = marnasnet_e(init_channels=init_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError("%s is not included" % args.arch)
    
    model.eval()
    input_tensor = torch.zeros(batch_size, init_channels, window_size)
    
    convert_model(model, input_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert each model to [CPU, GPU, NNAPI].')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    # convert
    parser.add_argument('--arch', type=str, default='mobilenet_v2', help='which architecture to use')
    parser.add_argument('--save', type=str, default='', help='saved path name')
    args = parser.parse_args()
    main(args)