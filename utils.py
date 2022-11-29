import glob
import os
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
sns.set()


def save_ckpt(model, optimizer, scheduler, epoch, model_path):
    torch.save({'model':model.state_dict(), 
                'epoch': epoch,
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict()}, model_path)


def restore_ckpt(model, optimizer, scheduler, model_path, location):
    state = torch.load(model_path, map_location=f'cuda:{location}')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    epoch = state['epoch']
    return epoch


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, location):
    model.load_state_dict(torch.load(model_path, map_location=f'cuda:{location}'), strict=True)


class PolicyHistory(object):

    def __init__(self, op_names, save_dir, n_class):
        self.op_names = op_names
        self.save_dir = save_dir
        self._initialize(n_class)

    def _initialize(self, n_class):
        self.history = []
        # [{m:[], w:[]}, {}]
        for i in range(n_class):
            self.history.append({'magnitudes': [],
                                'weights': [],
                                'var_magnitudes': [],
                                'var_weights': []})

    def add(self, class_idx, m_mu, w_mu, m_std, w_std):
        if not isinstance(m_mu, list):  # ugly way to bypass batch with single element
            return
        self.history[class_idx]['magnitudes'].append(m_mu)
        self.history[class_idx]['weights'].append(w_mu)
        self.history[class_idx]['var_magnitudes'].append(m_std)
        self.history[class_idx]['var_weights'].append(w_std)

    def save(self, class2label=None):
        path = os.path.join(self.save_dir, 'policy')
        vis_path = os.path.join(self.save_dir, 'vis_policy')
        os.makedirs(path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        header = ','.join(self.op_names)
        for i, history in enumerate(self.history):
            k = i if class2label is None else class2label[i]
            np.savetxt(f'{path}/policy{i}({k})_magnitude.csv',
                       history['magnitudes'], delimiter=',', header=header, comments='')
            np.savetxt(f'{path}/policy{i}({k})_weights.csv',
                       history['weights'], delimiter=',', header=header, comments='')
            np.savetxt(f'{vis_path}/policy{i}({k})_var_magnitude.csv',
                       history['var_magnitudes'], delimiter=',', header=header, comments='')
            np.savetxt(f'{vis_path}/policy{i}({k})_var_weights.csv',
                       history['var_weights'], delimiter=',', header=header, comments='')

    def plot(self):
        PATH = self.save_dir
        mag_file_list = glob.glob(f'{PATH}/policy/*_magnitude.csv')
        weights_file_list = glob.glob(f'{PATH}/policy/*_weights.csv')
        n_class = len(mag_file_list)

        f, axes = plt.subplots(n_class, 2, figsize=(15, 5*n_class))

        for i, file in enumerate(mag_file_list):
            df = pd.read_csv(file).dropna()
            x = range(0, len(df))
            y = df.to_numpy().T
            axes[i][0].stackplot(x, y, labels=df.columns, edgecolor='none')
            axes[i][0].set_title(file.split('/')[-1][:-4])

        for i, file in enumerate(weights_file_list):
            df = pd.read_csv(file).dropna()
            x = range(0, len(df))
            y = df.to_numpy().T
            axes[i][1].stackplot(x, y, labels=df.columns, edgecolor='none')
            axes[i][1].set_title(file.split('/')[-1][:-4])

        axes[-1][-1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.2), fancybox=True, shadow=True, ncol=10)
        plt.savefig(f'{PATH}/policy/schedule.png')

        f, axes = plt.subplots(1, 1, figsize=(7,5))

        frames = []
        for i, file in enumerate(mag_file_list):
            df = pd.read_csv(file).dropna()
            df['class'] = file.split('/')[-1][:-4].split('_')[0]
            frames.append(df.tail(1))

        df = pd.concat(frames)
        df.set_index('class').plot(ax=axes, kind='bar', stacked=True, legend=False, rot=90, fontsize=8)
        axes.set_ylabel("magnitude")
        plt.savefig(f'{PATH}/policy/magnitude_by_class.png')
        
        f, axes = plt.subplots(1, 1, figsize=(7,5))
        frames = []
        for i, file in enumerate(weights_file_list):
            df = pd.read_csv(file).dropna()
            df['class'] = file.split('/')[-1][:-4].split('_')[0].split('(')[1][:-1]
            frames.append(df.tail(1))

        df = pd.concat(frames)   
        df.set_index('class').plot(ax=axes, kind='bar', stacked=True, legend=False, rot=90, fontsize=8)
        axes.set_ylabel("probability")
        axes.set_xlabel("")
        plt.savefig(f'{PATH}/policy/probability_by_class.png')
        
        return f


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1.-drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(True)
