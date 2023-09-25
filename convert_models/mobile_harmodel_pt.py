import argparse
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# if torch.__version__ != '1.6.0' or torch.__version__ =='1.6.0+cu101':
#     raise ValueError("PyTorch version should be 1.6.0")

from thop import profile

from data_providers import *
from utils import *
from models import *


def main(args):
    if len(args.save) < 1:
        args.save = 'mobile_pt/{}'.format(args.arch)
    create_exp_dir(args.save, scripts_to_save=None)
    
    # dataset
    if args.dataset == 'uci':
        dataset = UCIHARDataProvider(data_path='', train_batch_size=512, test_batch_size=512, valid_size=None)
        args.dataset = 'UCI_HAR'
    elif args.dataset == 'opp':
        dataset = OPPDataProvider(data_path='', train_batch_size=512, test_batch_size=512, valid_size=None)
        args.dataset = 'OPPORTUNITY'
    elif args.dataset == 'kar':
        dataset = KUHARDataProvider(data_path='', train_batch_size=512, test_batch_size=512, valid_size=None)
        args.dataset = 'KUHAR'
    elif args.dataset == 'uni':
        dataset = UniMiBDataProvider(data_path='', train_batch_size=512, test_batch_size=512, valid_size=None)
        args.dataset = 'UniMiB-SHAR'
    elif args.dataset == 'wis':
        dataset = WISDMDataProvider(data_path='', train_batch_size=512, test_batch_size=512, valid_size=None)
        args.dataset = 'WISDM'
    else:
        raise ValueError("Unknown dataset type")
    
    init_channels, window_size = dataset.data_shape
    NUM_CLASSES = dataset.n_classes
    batch_size = args.batch_size

    # set device    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    
    model.to(device)
    model.eval()
    
    macs, param_size = profile(model, inputs=(input_tensor, ))
    print("Param Size = %fMB", param_size/1e6)
    print("FLOPs = %fMB", macs / 1e6)
    
    example = torch.randn(1, init_channels, window_size)
    traced_script_module = torch.jit.trace(model, example)
    model_path = os.path.join(args.save, f'{args.arch}/{args.dataset}.pt')
    torch.jit.save(traced_script_module, model_path)

# optimized_scripted_module = optimize_for_mobile(traced_script_module)
# torch.jit.save(optimized_scripted_module, 'UCI_HAR.pt')
# exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter("UCI_HAR.ptl")
# torch.jit.save(optimized_scripted_module, 'OPPORTUNITY.pt')
# exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter("OPPORTUNITY.ptl")
# exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter("model_lite.ptl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create model of each vision model for Mobile Device.')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='RTWCNN', help='which architecture to use')
    parser.add_argument('--save', type=str, default='', help='saved path name')
    # parser.add_argument('--config_file', type=str, default='harmodel_spec.csv', help='path to config file.')
    # parser.add_argument('--num-runs', type=int, default=100,
    #                     help='number of runs to compute average forward timing. default is 100')
    # parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    # parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)