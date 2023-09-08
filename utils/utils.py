import os
import shutil
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
      os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
      os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)


def calculate_model_size(model):
  param_size = 0
  for param in model.parameters():
      param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
      buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  
  return size_all_mb