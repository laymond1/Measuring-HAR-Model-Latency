import argparse
import os
import csv
import torch
import fvcore.nn as fnn

from thop import profile

from utils.utils import AverageMeter

from data_providers import *
from models import *


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
    # device = torch.device("cuda:0" if args.device == 'gpu' else "cpu")
    # create input dummy data
    input_tensor = torch.randn(batch_size, init_channels, window_size) #.to(device)
    
    # RTCNN
    if args.arch == 'RTCNN':
        model = RTCNN(flat_size=None, init_channels=init_channels, num_classes=NUM_CLASSES, acc_num=None) # TODO: fix this
    elif args.arch == 'RTWCNN':
        model = RTWCNN(init_channels=init_channels, num_classes=NUM_CLASSES, segment_size=window_size)
    # LSTM
    elif args.arch == 'HARLSTM':
        model = HARLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'HARBiLSTM':
        model = HARBiLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'HARConvLSTM':
        model = HARConvLSTM(init_channels=init_channels, num_classes=NUM_CLASSES)
    # TSC
    elif args.arch == 'ResNetTSC':
        model = ResNetTSC(init_channels=init_channels, num_classes=NUM_CLASSES)
    elif args.arch == 'FCNTSC':
        model = FCNTSC(init_channels=init_channels, num_classes=NUM_CLASSES)
    else:
        raise ValueError("%s is not included" % args.arch)
    
    # model.to(device)
    model.eval()

    flops, params = fnn.FlopCountAnalysis(model, input_tensor), fnn.parameter_count(model)
    print("Flops: %f, Param size: %fMB" % (flops.total(), params['']))

    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    print("Flops: %f, Param size: %fMB" % (macs/1e6, params/1e6))
    
    # CSV 파일에 실험 결과 저장
    # df = pd.read_csv('model_spec.csv')
    filename = args.config_file

    with open(filename, mode='a', newline='') as f:
        list_data = [args.dataset, args.arch, params/1e6, macs/1e6]
        # Pass the CSV  file object to the writer() function
        writer = csv.writer(f)
        writer.writerow(list_data)  

    print("Experiment results saved to", filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute latency of each vision model for 1D data.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='RTWCNN', help='which architecture to use')
    parser.add_argument('--config_file', type=str, default='harmodel_spec.csv', help='path to config file.')
    # parser.add_argument('--num-runs', type=int, default=100,
    #                     help='number of runs to compute average forward timing. default is 100')
    # parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)