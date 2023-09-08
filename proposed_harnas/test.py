import os
import sys
import time
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from datetime import datetime
import sklearn.metrics as metrics
from thop import profile
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from seaborn import heatmap

import genotypes
from model import NetworkHAR

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('./data_providers/'))))
from data_providers import UCIHARDataProvider, OPPDataProvider, KUHARDataProvider, UniMiBDataProvider, WISDMDataProvider


parser = argparse.ArgumentParser("DNAS-based HARNAS Architecture Test")
parser.add_argument('--dataset', type=str, default='wis', choices=['uci', 'opp', 'kar', 'uni', 'wis'], help='dataset to use')
parser.add_argument('--data_path', type=str, default='', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='', help='path of pretrained model')
parser.add_argument('--classifier', type=str, default='LSTM', help='type of classifier')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

if args.save == 'EXP':
    args.save = 'result_pr/{}/test-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if args.dataset == 'uci':
	dataset = UCIHARDataProvider(data_path=args.data_path, 
								 train_batch_size=args.batch_size, test_batch_size=args.batch_size, 
								 valid_size=None)
elif args.dataset == 'opp':
	dataset = OPPDataProvider(data_path=args.data_path, 
							  train_batch_size=args.batch_size, test_batch_size=args.batch_size, 
							  valid_size=None)
elif args.dataset == 'kar':
	dataset = KUHARDataProvider(data_path=args.data_path, 
								train_batch_size=args.batch_size, test_batch_size=args.batch_size, 
								valid_size=None)
elif args.dataset == 'uni':
	dataset = UniMiBDataProvider(data_path=args.data_path, 
								 train_batch_size=args.batch_size, test_batch_size=args.batch_size, 
								 valid_size=None)
elif args.dataset == 'wis':
	dataset = WISDMDataProvider(data_path=args.data_path, 
								train_batch_size=args.batch_size, test_batch_size=args.batch_size, 
								valid_size=None)
else:
	raise ValueError("Unknown dataset type")

# args.arch = 'EANAS' # TODO remove this line
init_channels, window_size = dataset.data_shape
NUM_CLASSES = dataset.n_classes
layers = args.layers
model_path = args.model_path
classifier = args.classifier


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    ckpt = torch.load(model_path)
    model = NetworkHAR(init_channels, NUM_CLASSES, layers, genotype, classifier)
    model = model.cuda()
    model.load_state_dict(ckpt['model'].state_dict())

    logging.info("Architecture = %s", genotype)
    input = torch.randn(1, init_channels, window_size, 1).cuda()
    macs, params = profile(model, inputs=(input, ), verbose=False)
    logging.info("Flops: %f, Param size: %fMB", macs / 1e6, params / 1e6)

    targets_cumulative = dataset.Ytest
    criterion = nn.CrossEntropyLoss()

    start = datetime.now()

    test_loss, top_classes = infer(dataset.test, model, criterion)

    equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
    accuracy = np.mean(equals)

    f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
    f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
    
    end = datetime.now()
    time = (end - start).total_seconds()

    logging.info('test_acc %.3f', accuracy)
    logging.info('macro-test_f1 %.3f', f1macro)
    logging.info('Weighted F1-score: %.3f', f1score)

    # Visualization of confusion matrix
    targets, target_names = dataset.Ytest, dataset.target_names
    mat = confusion_matrix(targets, top_classes, normalize='true')
    df_cm = pd.DataFrame(mat, index=target_names,columns=target_names)

    plt.figure(10,figsize=(15,12))
    heatmap(df_cm,annot=True,fmt='.2f',cmap='Purples')
    plt.savefig(os.path.join(args.save, 'confusion_matrix.png'))
    #   plt.show()

    conf_mat = classification_report(dataset.Ytest, top_classes, target_names=target_names, digits=4)
    logging.info(conf_mat)
    
    # save
    ckpt = {
        'model': model,
        'accuracy': accuracy,
        'f1macro': f1macro,
        'f1score': f1score,
        'conf_mat': conf_mat,
        'flops': macs / 1e6,
        'params': params / 1e6,
    }
    with open(os.path.join(args.save, 'ckpt_best.pkl'), 'wb') as file:
        pickle.dump(ckpt, file)
  

def infer(test_queue, model, criterion):
    losses = utils.AverageMeter()
    model.eval()  # Setup network for evaluation

    top_classes = []
    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            
            input = input.view(-1, input.size(1), input.size(2), 1).cuda()
            target = target.cuda()

            if model.classifier == 'LSTM':
                output, h = model(input)
            else:
                output = model(input)
            
            val_loss = criterion(output, target)
            n = input.size(0)
            losses.update(val_loss.item(), n)

            top_p, top_class = output.topk(1, dim=1)
            top_classes.extend([top_class.item() for top_class in top_class.cpu()])
    
    return losses.avg, top_classes


if __name__ == '__main__':
    main() 

