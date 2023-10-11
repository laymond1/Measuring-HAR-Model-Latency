import argparse
import os
import sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import create_exp_dir
from data_providers import *
from models import *

# if '1.6.0' not in torch.__version__:
#     raise ValueError("PyTorch version should be 1.6.0")

def convert_model(net, input_tensor):
    # convert model to torchscript
    traced_script_module = torch.jit.trace(net, input_tensor)
    model_path = os.path.join(args.save, f'{args.block_name}-k{args.kernel_size}.pt')
    torch.jit.save(traced_script_module, model_path)
    print(f"CPU model saved at : {model_path}")


def main(args):
    if len(args.save) < 1:
        args.save = 'mobile_blocks_pt/{}'.format(args.dataset)
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
    block.eval()
    input_tensor = torch.randn(batch_size, args.input_channels, window_size)
    # input_tensor = torch.zeros(batch_size, init_channels, window_size)
    
    convert_model(block, input_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert each model to [CPU, GPU, NNAPI].')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--block_name', type=str, default='GTSResConvBlock', 
                        choices=['LSTMBlock', 'BiLSTMBlock', 'GTSResConvBlock'], help='which architecture to use')
    parser.add_argument('--kernel_size', type=int, default=0)
    parser.add_argument('--input_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--skip_op', type=str, default='None', choices=['None', 'pool', 'identity'])
    parser.add_argument('--num_layers', type=int, default=1, help='the number of layers')
    parser.add_argument('--se_ratio', type=float, default=0.0, help='the ratio of squeeze and excitation')
    # convert
    parser.add_argument('--save', type=str, default='', help='saved path name')
    args = parser.parse_args()
    main(args)