import os
import sys
import time
import glob
import numpy as np
from numpy import random
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/fasterdatasets/cifar-10', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or cifar100')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='test', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# [START ORIGINAL CODE - WILL BE REPLACED]
# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
# [END ORIGINAL CODE]

# [START NEW CODE - SINGLE EXPERIMENT FOLDER]
# Modified to create only one experiment folder
args.save = 'eval-{}'.format(args.save)  # Fixed experiment folder name
# Create experiment directory only if it doesn't exist
if not os.path.exists(args.save):
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
# [END NEW CODE]

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# [START ORIGINAL CODE - WILL BE REPLACED]
#fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# [END ORIGINAL CODE]

# [START NEW CODE
# Add timestamp to log filename to prevent overwriting
log_filename = 'log_{}.txt'.format(time.strftime("%Y%m%d-%H%M%S"))
fh = logging.FileHandler(os.path.join(args.save, log_filename))
# [END NEW CODE]
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
    CIFAR_CLASSES = 10
elif args.dataset == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    # fix seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info('genotype = %s', genotype)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info(model)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    if args.dataset == 'cifar10':
        dataset_class = dset.CIFAR10
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    elif args.dataset == 'cifar100':
        dataset_class = dset.CIFAR100
        train_transform, valid_transform = utils._data_transforms(args)

    train_data = dataset_class(root=args.data, train=True, download=False, transform=train_transform)
    valid_data = dataset_class(root=args.data, train=False, download=False, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs),
                                                           eta_min=args.learning_rate_min)

    best_valid_acc = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc)

        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                # [START ORIGINAL CODE - WILL BE REPLACED]
                # utils.save(model, os.path.join(args.save, 'best_weights.pt'))
                # [END ORIGINAL CODE]
                
                # [START NEW CODE - TIMESTAMPED WEIGHTS]
                # TEMPORARILY DISABLED: Only save at the end - store state temporarily
                best_model_state = model.state_dict().copy()
                # [END NEW CODE]
            if epoch % 50 == 0:  # Log every 50 epochs to reduce output
                logging.info('best_valid_acc %f', best_valid_acc)

        # [START ORIGINAL CODE - WILL BE REPLACED]
        # utils.save(model, os.path.join(args.save, 'weights.pt'))
        # [END ORIGINAL CODE]
        
        # [START NEW CODE - TIMESTAMPED WEIGHTS]
        # TEMPORARILY DISABLED: Only save at the end
        # utils.save(model, os.path.join(args.save, 'weights_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"))))
        # [END NEW CODE]
    
    # [NEW CODE] Save only the final best model and final model
    if best_model_state is not None:
        # Save the best model
        model.load_state_dict(best_model_state)  # Load best state
        utils.save(model, os.path.join(args.save, 'best_weights_final.pt'))
    
    # Save the final model after all epochs
    utils.save(model, os.path.join(args.save, 'weights_final.pt'))
    logging.info('Final best validation accuracy: %f', best_valid_acc)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # input = Variable(input).cuda()
        # target = Variable(target).cuda(async=True)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()

