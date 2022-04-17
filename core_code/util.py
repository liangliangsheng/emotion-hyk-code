import os
import torch
from pathlib import Path
import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_path):
    if not os.path.exists('./model'):
        os.makedirs('./model')
    torch.save(state, save_path)


def load_model(model, path, resume):
    checkpoint = torch.load(path)
    pretrained_state_dict = checkpoint['state_dict']
    if resume:
        epoch = checkpoint['epoch']
        acc_video = checkpoint['accuracy_video']
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(pretrained_state_dict)
        return epoch, acc_video, model
    else:
        model_state_dict = model.state_dict()
        for key in pretrained_state_dict:
            if (key == 'module.fc.weight') | (key == 'module.fc.bias'):
                pass
            else:
                model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict)
        model = torch.nn.DataParallel(model).cuda()
        return model


def time_now():
    return datetime.datetime.now().strftime('%d-%h-%Y-%H-%M-%S')


class Logger(object):
    def __init__(self, log_dir, title, time, aggregate_mode, args=False):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
        self.title = title
        self.log_file = '{:}/{:}_{:}_{:}.txt'.format(self.log_dir, title, aggregate_mode, time)
        self.file_writer = open(self.log_file, 'a')

        if args:
            for key, value in vars(args).items():
                self.print('  [{:18s}] : {:}'.format(key, value))

    def print(self, string, fprint=True):
        print(string)
        if fprint:
            self.file_writer.write('{:}\n'.format(string))
            self.file_writer.flush()

    def write(self, string):
        self.file_writer.write('{:}\n'.format(string))
        self.file_writer.flush()
