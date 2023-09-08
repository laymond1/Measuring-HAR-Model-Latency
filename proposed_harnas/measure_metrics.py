import argparse
import os
import csv
import sys
import torch
import fvcore.nn as fnn

from thop import profile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import proposed_harnas.genotypes as genotypes
from proposed_harnas.model import NetworkHAR
from proposed_harnas.pro_utils import AverageMeter

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
    stem_multiplier = dataset.stem_multiplier
    multiplier = dataset.multiplier
    NUM_CLASSES = dataset.n_classes
    batch_size = args.batch_size
    classifier = 'LSTM'
    layers = 1
    
    # set device    
    # device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    # create input dummy data
    input_tensor = torch.randn(batch_size, init_channels, window_size, 1) #.to(device)
    
    # measure the latency of model
    genotype = eval("genotypes.{}".format(args.arch))
    model = NetworkHAR(init_channels, NUM_CLASSES, layers, genotype, classifier)
    # model.to(device)
    model.eval()

    flops, params = fnn.FlopCountAnalysis(model, input_tensor), fnn.parameter_count(model)
    print("Flops: %f, Param size: %fMB" % (flops.total(), params['']))

    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    print("Flops: %f, Param size: %fMB" % (macs/1e6, params/1e6))
    
    size_all_mb = calculate_model_size(model)
    print('model size: {:.3f}MB'.format(size_all_mb))
    
    # CSV 파일에 실험 결과 저장
    # df = pd.read_csv('harnas_spec.csv')
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.arch, params/1e6, macs/1e6, size_all_mb]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute latency of each vision model for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='./', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='OPPDNAS', help='which architecture to use')
    parser.add_argument('--config_file', type=str, default='harnas_spec.csv', help='path to config file.')
    # parser.add_argument('--num-runs', type=int, default=100,
    #                     help='number of runs to compute average forward timing. default is 100')
    # parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)