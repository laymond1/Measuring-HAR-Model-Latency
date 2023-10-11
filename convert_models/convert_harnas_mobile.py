import argparse
import os
import sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import create_exp_dir
from data_providers import *

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
        args.save = 'mobile_models_pt/{}'.format(args.dataset)
    create_exp_dir(args.save, scripts_to_save=None)
    
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
    layers = 1
    
    # RLNAS
    if args.arch == 'RLNAS':
        from rl_harnas.model import RLNASModel
        model = RLNASModel(C=init_channels, num_classes=NUM_CLASSES)
        input_tensor = torch.zeros(batch_size, init_channels, window_size)
    # EANAS
    elif args.arch == 'EANAS':
        import ea_harnas.genotypes as genotypes
        from ea_harnas.micro_models import NetworkHAR
        genotype = eval("genotypes.{}".format(args.arch))
        model = NetworkHAR(init_channels, NUM_CLASSES, layers, genotype)
        input_tensor = torch.zeros(batch_size, init_channels, window_size, 1)
    # DNAS
    elif args.arch == 'DNAS':
        import proposed_harnas.genotypes as genotypes
        from proposed_harnas.model import NetworkHAR
        genotype = eval("genotypes.{}".format(args.arch))
        model = NetworkHAR(init_channels, NUM_CLASSES, layers, genotype, classifier)
        input_tensor = torch.zeros(batch_size, init_channels, window_size, 1)
    else:
        raise ValueError("%s is not included" % args.arch)
    
    model.eval()
    convert_model(model, input_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert each model to [CPU, GPU, NNAPI].')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--data_path', type=str, default='./', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    # convert
    parser.add_argument('--arch', type=str, default='RLNAS', help='which architecture to use')
    parser.add_argument('--save', type=str, default='', help='saved path name')
    args = parser.parse_args()
    main(args)