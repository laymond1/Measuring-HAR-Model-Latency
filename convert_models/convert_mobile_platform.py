import argparse
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from data_providers import *
from models import * # RTCNN RTWCNN HARLSTM HARBiLSTM HARConvLSTM ResNetTSC FCNTSC


def convert_model(net, input_tensor):
    net = net.eval()
    
    # 1. CPU
    model_script = torch.jit.script(net)
    model_cpu = optimize_for_mobile(model_script)
    model_cpu._save_for_lite_interpreter(f"mobile_pt/{args.arch}/model_cpu.ptl")
    print("CPU model saved at : mobile_pt/model_cpu.ptl")
    
    # 2. GPU(Vulkan)
    model_vulkan = optimize_for_mobile(model_script, backend="Vulkan")
    model_vulkan._save_for_lite_interpreter(f"mobile_pt/{args.arch}/model_vulkan.ptl")
    print("GPU(Vulkan) model saved at : mobile_pt/model_vulkan.ptl")
    
    # 3. NNAPI
    # input_float = torch.zeros(1, 3, 128)
    # input_tensor = input_float
    # input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    input_tensor = input_tensor.contiguous()
    with torch.no_grad():
        traced = torch.jit.trace(net, input_tensor)
        model_nnapi = torch.backends._nnapi.prepare.convert_model_to_nnapi(traced, input_tensor)
        model_nnapi._save_for_lite_interpreter(f"mobile_pt/{args.arch}/model_nnapi.ptl")
    print("NNAPI model saved at : mobile_pt/model_nnapi.ptl")


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
    
    model.eval()
    input_tensor = torch.zeros(batch_size, init_channels, window_size)
    
    convert_model(model, input_tensor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert each model to [CPU, GPU, NNAPI].')
    parser.add_argument('--dataset', type=str, default='uci', choices=['uci', 'opp', 'kar', 'uni', 'wis'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size. default is 1')
    parser.add_argument('--arch', type=str, default='RTWCNN', help='which architecture to use')
    parser.add_argument('--hardware', type=str, default='pc', choices=['pc', 'nano'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    args = parser.parse_args()
    main(args)