
import argparse
import os
import csv
import sys
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.utils import AverageMeter

from data_providers import *
from models.vision import *

def main(args):
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
    
    # set device    
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    # device = torch.device("%s" % args.device)
    
    # create input dummy data
    input_tensor = torch.randn(batch_size, init_channels, window_size).to(device)
    
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
    else:
        raise ValueError("%s is not included" % args.arch)
    
    model.to(device)
    model.eval()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # warm-up 실행
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # 
    latency = AverageMeter()
    with torch.no_grad():
        # for i in tqdm(range(args.num_runs)):
        for i in range(args.num_runs):
            
            starter.record()            
            model(input_tensor)
            ender.record()
            
            torch.cuda.synchronize()
            latency.update(starter.elapsed_time(ender)) # miliseconds
        print("%s: %f" % (args.arch, latency.avg))
    
    # CSV 파일에 실험 결과 저장
    # df = pd.read_csv('vision.csv')
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.batch_size, args.arch, args.hardware, args.device, latency.avg]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute latency of each vision model for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='mobilenet_v2', help='which architecture to use')
    parser.add_argument('--config_file', type=str, default='vision.csv', help='path to config file.')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='number of runs to compute average forward timing. default is 100')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)