from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import time
import copy
import higher
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils

from adaptive_augmentor import AdaAug
from networks import get_model
from networks.projection import Projection,Augment_decision
from config import get_search_divider
from config import get_warmup_config
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler
from load_data import CIFAR10
from dataset import get_dataloaders, get_num_class, get_label_name, get_dataset_dimension,CutoutDefault

parser = argparse.ArgumentParser("ada_aug")
parser.add_argument('--dataroot', type=str, default='./', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='name of dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.400, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='./search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=1, help='portion of training data')
parser.add_argument('--proj_learning_rate', type=float, default=1e-2, help='learning rate for h')
parser.add_argument('--proj_weight_decay', type=float, default=1e-3, help='weight decay for h]')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--model_name', type=str, default='wresnet40_2', help="mode _name")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--k_ops', type=int, default=1, help="number of augmentation applied during training")
parser.add_argument('--threshold', type=float, default=0.5, help="the threshold for augmentation")
parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
parser.add_argument('--search_freq', type=float, default=1, help='exploration frequency')
parser.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")

args = parser.parse_args()


def main():
    if not torch.cuda.is_available():
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    utils.reproducibility(args.seed)
    
   
    #  dataset settings
    n_class = get_num_class(args.dataset)
    sdiv = get_search_divider(args.model_name)
    class2label = get_label_name(args.dataset, args.dataroot)
    
    train_queue, valid_queue, search_queue, test_queue = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        args.dataroot, args.cutout, args.cutout_length,
        split=args.train_portion, split_idx=0, target_lb=-1,
        search=True, search_divider=sdiv)
   
    #  model settings
    gf_model = get_model(model_name=args.model_name, num_class=n_class,
        use_cuda=True, data_parallel=False)
    
    h_model = Projection(in_features=gf_model.fc.in_features,
        n_layers=args.n_proj_layer, n_hidden=128).cuda()

    #  training settings
    gf_optimizer = torch.optim.SGD(
        gf_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gf_optimizer,
        float(args.epochs), eta_min=args.learning_rate_min)

    h_optimizer = torch.optim.Adam(
        h_model.parameters(),
        lr=args.proj_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.proj_weight_decay)
    
    m, e = get_warmup_config(args.dataset)
    scheduler = GradualWarmupScheduler(
            gf_optimizer,
            multiplier=m,
            total_epoch=e,
            after_scheduler=scheduler)
    
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    #  AdaAug settings
    after_transforms = train_queue.dataset.after_transforms
    adaaug_config = {'sampling': 'prob',
                    'k_ops': 1,
                    'delta': 0.0,
                    'temp': 1.0,
                    'search_d': get_dataset_dimension(args.dataset),
                    'target_d': get_dataset_dimension(args.dataset)}

    adaaug = AdaAug(after_transforms=after_transforms,
        n_class=n_class,
        gf_model=gf_model,
        h_model=h_model,
        save_dir=args.save,
        config=adaaug_config)

    #  Start training
    start_time = time.time()
    best_acc1 = 0
    for epoch in range(args.epochs):
        print('epoch',epoch)
        # searching
        train_acc, train_obj = train(train_queue, search_queue, gf_model, adaaug,
            criterion, gf_optimizer, args.grad_clip, h_optimizer, epoch, args.search_freq
            )
        print('train_acc',train_acc)
        # validation
        valid_acc, valid_obj = infer(test_queue, gf_model, criterion)
        print('valid_acc',valid_acc)
        best_acc1 = max(valid_acc, best_acc1)
        print('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
       
        scheduler.step()

        #utils.save_model(gf_model, os.path.join(args.save, 'gf_weights_cifar10.pt'))
        #tils.save_model(h_model, os.path.join(args.save, 'h_weights_meat_cifar10.pt'))

    end_time = time.time()



def train(train_queue, valid_queue, gf_model, adaaug, criterion, gf_optimizer,
            grad_clip, h_optimizer, epoch, search_freq):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)        
        
        if step % search_freq == 0 :
           h_optimizer.zero_grad()
           with higher.innerloop_ctx(gf_model, gf_optimizer) as (meta_model, diffopt):
             adaaug.gf_model = meta_model
             aug_images = adaaug(input, mode='exploit')
             logits = meta_model(aug_images)
             loss = criterion(logits, target) 
             
             nn.utils.clip_grad_norm_(meta_model.parameters(), grad_clip)
             diffopt.step(loss)
            
             torch.cuda.empty_cache()
             input_search, target_search = next(iter(valid_queue))
             input_search =input_search.cuda(non_blocking=True) 
             target_search = target_search.cuda(non_blocking=True)
             logits = meta_model(input_search)
             loss = criterion(logits, target_search)
             loss.backward()
      
           h_optimizer.step()
           adaaug.gf_model = copy.deepcopy(gf_model)
        
        aug_image = adaaug(input, mode='exploit')     
        gf_model.train()
        gf_optimizer.zero_grad()
        logits = gf_model(aug_image)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(gf_model.parameters(), grad_clip)
        gf_optimizer.step()
            
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.detach().item(), n)
        top1.update(prec1.detach().item(), n)
        top5.update(prec5.detach().item(), n)
       
        
    return top1.avg, objs.avg


def infer(valid_queue, gf_model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    gf_model.eval()

    with torch.no_grad():
        for input, target in valid_queue:
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = gf_model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.detach().item(), n)
            top1.update(prec1.detach().item(), n)
            top5.update(prec5.detach().item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
