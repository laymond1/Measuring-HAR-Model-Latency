
import argparse
import os
import csv
import sys
import math
import torch
import torch.nn as nn
import pandas as pd

from thop import profile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rl_harnas.genotypes as genotypes
from rl_harnas.model_generator import get_spec

from rl_harnas.nas_utils import AverageMeter

from data_providers import *


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
    NUM_CLASSES = dataset.n_classes
    batch_size = args.batch_size
    classifier = 'LSTM'
    
    # set device    
    device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    # device = torch.device("%s" % args.device)
    
    # create input dummy data
    input_tensor = torch.randn(batch_size, init_channels, window_size).to(device)

    # measure the latency of model
    genotype = eval("genotypes.{}".format(args.arch))
    model = get_spec(genotype)
    model = model.write(NUM_CLASSES, init_channels, window_size, batch_size, classifier)
    model.to(device)
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    # measure
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
        
    # save the data to the CSV file
    # df = pd.read_csv('harnas.csv')
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.batch_size, args.arch, args.hardware, args.device, latency.avg]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Compute latency of each operation.')
    # parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    # parser.add_argument('--arch', type=str, default='RLNAS', help='which architecture to use')
    # parser.add_argument('--config-file', type=str, help='path to config file.')
    # parser.add_argument('--num-runs', type=int, default=100,
    #                     help='number of runs to compute average forward timing. default is 100')
    parser = argparse.ArgumentParser(description='Compute latency of each vision model for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='../', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='RLNAS', help='which architecture to use')
    parser.add_argument('--config_file', type=str, default='harnas.csv', help='path to config file.')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='number of runs to compute average forward timing. default is 100')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)