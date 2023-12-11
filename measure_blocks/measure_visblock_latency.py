
import argparse
import os
import csv
import sys
import torch
import pandas as pd
from tqdm import tqdm
from thop import profile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.utils import AverageMeter

from data_providers import *
from models.vision.blocks import *
from models.vision import create_block


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
    batch_size = args.batch_size
    
    # set device    
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    
    # create input dummy data
    input_tensor = torch.randn(batch_size, args.input_channels, window_size).to(device)
    
    # 1. block: [MB(w/ SE), FusedMB(w/ SE), Conv, SepConv, MB, MB(w/o SE), ResBlock, ResBotneck, ShuffleMB]
    cnf = BlockConfig(
            conv_op=eval(args.block_name),
            repeats=args.num_layers, 
            kernel=args.kernel_size, stride=1, 
            input_channels=args.input_channels, out_channels=args.out_channels, 
            skip_op=args.skip_op,
            se_ratio=args.se_ratio
        )
    block = create_block(block_name=args.block_name, cnf=cnf)
    # 2. kernel size: [1, 3, 5, 7, 9]
    kernel_size_list = [1, 3, 5, 7, 9]
    # 3. out_channels: [16, 32, 64, 128]
    out_channels_list = [16, 32, 64, 128]
    # 4. num_layers: [1, 2, 3, 4, 5]
    num_layers_list = [1, 2, 3, 4, 5]
    
    block.to(device)
    block.eval()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = block(input_tensor)

    # 
    latency = AverageMeter()
    with torch.no_grad():
        # for i in tqdm(range(args.num_runs)):
        for i in range(args.num_runs):
            
            starter.record()            
            block(input_tensor)
            ender.record()
            
            torch.cuda.synchronize()
            latency.update(starter.elapsed_time(ender)) # miliseconds
        print("%s: %f" % (args.block_name, latency.avg))
    
    # save CSV
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.batch_size, window_size, args.block_name, args.kernel_size, args.input_channels, args.out_channels, args.num_layers, args.hardware, args.device, latency.avg]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute latency of each vision block for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--block_name', type=str, default='ShuffleBlock', 
                        choices=['ConvBlock', 'SeparableConvBlock', 'MBConvBlock', 'ResConvBlock', 'ShuffleBlock'], help='which architecture to use')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--input_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--skip_op', type=str, default='None', choices=['None', 'pool', 'identity'])
    parser.add_argument('--num_layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--se_ratio', type=float, default=0.0, help='the ratio of squeeze and excitation')
    parser.add_argument('--config_file', type=str, default='visblocks.csv', help='path to config file.')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='number of runs to compute average forward timing. default is 100')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)