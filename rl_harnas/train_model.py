import os
import sys
import time
import glob
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import  torch.nn as nn
import torch.backends.cudnn as cudnn

import genotypes
from model_generator import get_spec
from data_providers import UCIHARDataProvider, OPPDataProvider, KUHARDataProvider, UniMiBDataProvider, WISDMDataProvider

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from nas_utils import *
else:
    # uses current package visibility
    from .nas_utils import *

import sklearn.metrics as metrics
from datetime import datetime

parser = argparse.ArgumentParser("RL-based HARNAS Architecture Training")
parser.add_argument('--dataset', type=str, default='opp', choices=['uci', 'opp', 'kar', 'uni', 'wis'], help='dataset to use')
parser.add_argument('--data_path', type=str, default='', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--early_stop', type=int, default=20, help='early stop')
parser.add_argument('--lr_schedule', type=str, default='cosine', help='lr schedule')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--classifier', type=str, default='LSTM', help='type of classifier')
parser.add_argument('--arch', type=str, default='', help='which genotype to use')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

if args.save == 'EXP':
  args.save = 'rl_result/{}/train-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

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

# args.arch = 'RLNAS' # TODO remove this line
init_channels, window_size = dataset.data_shape
NUM_CLASSES = dataset.n_classes
layers = args.layers
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
    model = get_spec(genotype)
    # density = self.get_density(model)
    model = model.write(NUM_CLASSES, init_channels, window_size, args.batch_size, args.classifier)
    model = model.cuda()
    
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    optimizer = torch.optim.SGD(model.parameters(),
                                args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # deleted AdamW optimizer
    
    targets_cumulative = dataset.Ytest
    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    
    start = datetime.now()

    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.early_stop, 
                                       verbose=False, 
                                       get_data=dataset.name(),
                                       path=args.save)

    if args.lr_schedule == 'step':
        lr_step = 100
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    lr_step)  # Learning rate scheduler to reduce LR every 100 epochs
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
        
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        
        train_acc, train_loss = train(dataset.train, model, criterion, optimizer)
        
        val_loss, top_classes = infer(dataset.test, model, val_criterion)
        
        equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
        accuracy = np.mean(equals)

        f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
        f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
        stopping_metric = f1score # be negative at stopping_metric
        # stopping_metric = val_loss # be positive at stopping_metric
        
        logging.info(
                'Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.3f}, f1: {:.3f}, M f1: {:.3f}, '
                'M: {:.4f}'.format(
                    epoch + 1, args.epochs, train_loss, val_loss, accuracy, f1score, f1macro,
                    stopping_metric))
        
        if args.early_stop:
            early_stopping((-stopping_metric), model, epoch)
            if early_stopping.early_stop:
                break
        if args.lr_schedule:
            scheduler.step()
            
    end = datetime.now()
    time = (end - start).total_seconds()
    logging.info('Best F1: %.3f, Train time: %.3f', early_stopping.best_score, time)
        

def train(train_queue, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    
    for step, (input, target) in enumerate(train_queue):
        input = input.view(-1, input.size(1), input.size(2), 1).cuda()
        target = target.cuda()

        optimizer.zero_grad()
        
        if model.classifier == 'LSTM':
            output, h = model(input)
        else:
            output = model(input)

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    
        prec1, _ = accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
        
    return top1.avg, losses.avg 


def infer(valid_queue, model, criterion):
    losses = AverageMeter()
    model.eval()  # Setup network for evaluation

    top_classes = []

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            
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

