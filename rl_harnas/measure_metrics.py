import argparse
import os
import csv
import sys
import torch
import fvcore.nn as fnn

from thop import profile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import rl_harnas.genotypes as genotypes
from rl_harnas.model_generator import get_spec

from utils.utils import AverageMeter, calculate_model_size

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
    
    input_tensor = torch.randn(batch_size, init_channels, window_size, 1)
    
    # measure the metrics of model
    genotype = eval("genotypes.{}".format(args.arch))
    model = get_spec(genotype)
    model = model.write(NUM_CLASSES, init_channels, window_size, batch_size, classifier)
    model.eval()

    flops, params = fnn.FlopCountAnalysis(model, input_tensor), fnn.parameter_count(model)
    print("Flops: %f, Param size: %fMB" % (flops.total(), params['']))

    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    print("Flops: %f, Param size: %fMB" % (macs/1e6, params/1e6))
    
    size_all_mb = calculate_model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))
    
    # save CSV
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.arch, params/1e6, macs/1e6, size_all_mb]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics of each vision model for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='./', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='RLNAS', help='which architecture to use')
    parser.add_argument('--config_file', type=str, default='harnas_spec.csv', help='path to config file.')
    args = parser.parse_args()
    main(args)