
import argparse
import os
import csv
import sys
import torch
import pandas as pd
from tqdm import tqdm
import fvcore.nn as fnn
from thop import profile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.utils import AverageMeter, calculate_model_size

from data_providers import *
from models.harblocks import *
from models import create_harblock


def main(args):
    # dataset
    if args.dataset == 'uci':
        dataset = UCIHARDataProvider(data_path=args.data_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'opp':
        dataset = OPPDataProvider(data_path=args.data_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'kar':
        dataset = KUHARDataProvider(data_path=args.data_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'uni':
        dataset = UniMiBDataProvider(data_path=args.data_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    elif args.dataset == 'wis':
        dataset = WISDMDataProvider(data_path=args.data_path, train_batch_size=args.batch_size, test_batch_size=args.batch_size, valid_size=None)
    else:
        raise ValueError("Unknown dataset type")
    
    init_channels, window_size = dataset.data_shape
    batch_size = args.batch_size
    
    # create input dummy data
    input_tensor = torch.randn(batch_size, args.input_channels, window_size)
    
    # 1. block: [LSTMBlock, BiLSTMBlock, GTSResConvBlock]
    if args.block_name == "GTSResConvBlock":
        cnf = GTSResBlockConfig(
            conv_op=eval(args.block_name),
            repeats=args.num_layers, 
            kernel=args.kernel_size, stride=1, 
            input_channels=args.input_channels, out_channels=args.out_channels,
            n_groups=16, # args.n_groups,
            first_grouped_conv=True, #args.first_grouped_conv,
            pool=True, # args.pool,
            skip_op=args.skip_op,
            se_ratio=args.se_ratio
        )
    else:
        cnf = BlockConfig(
                conv_op=eval(args.block_name),
                repeats=args.num_layers, 
                kernel=args.kernel_size, stride=1, 
                input_channels=args.input_channels, out_channels=args.out_channels, 
                skip_op=args.skip_op,
                se_ratio=args.se_ratio
            )
    block = create_harblock(block_name=args.block_name, cnf=cnf)
    # 2. kernel size: [1, 3, 5, 7, 9]
    kernel_size_list = [1, 3, 5, 7, 9]
    # 3. out_channels: [16, 32, 64, 128]
    out_channels_list = [16, 32, 64, 128]
    # 4. num_layers: [1, 2, 3, 4, 5]
    num_layers_list = [1, 2, 3, 4, 5]
    
    block.eval()
    
    flops, params = fnn.FlopCountAnalysis(block, input_tensor), fnn.parameter_count(block)
    print("Flops: %f, Param size: %fMB" % (flops.total(), params['']))

    macs, params = profile(block, inputs=(input_tensor, ), verbose=False)
    print("Flops: %f, Param size: %fMB" % (macs/1e6, params/1e6))
    
    size_all_mb = calculate_model_size(block)
    print('block size: {:.3f}MB'.format(size_all_mb))
    
    # save CSV
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.block_name, params/1e6, macs/1e6, size_all_mb]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metric of each vision block for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--block_name', type=str, default='GTSResConvBlock', 
                        choices=['LSTMBlock', 'BiLSTMBlock', 'GTSResConvBlock'], help='which architecture to use')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--input_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--skip_op', type=str, default='None', choices=['None', 'pool', 'identity'])
    parser.add_argument('--num_layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--se_ratio', type=float, default=0.0, help='the ratio of squeeze and excitation')
    parser.add_argument('--config_file', type=str, default='block_spec.csv', help='path to config file.')
    args = parser.parse_args()
    main(args)