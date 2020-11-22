from __future__ import division
import sys
import os
import os.path as osp
import random
import argparse

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from mmcv.runner import init_dist
from tensorboardX import SummaryWriter

from utils.distributed_utils import average_gradients, broadcast_params
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
import model.deeplab_vggbn as deeplab_vggbn
from model.discriminator import FCDiscriminator
from model.syncbn_layer import SyncBatchNorm2d
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.gta5bdd_dataset import GTA5BDDDataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.bdd_dataset import BDDDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './C-Driving/train/source'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './C-Driving/train/compound'
DATA_LIST_PATH_TARGET = './dataset/bdd_list/train/3domains.txt'
INPUT_SIZE_TARGET = '960,540'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'bdd100k'
SET = 'train'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument('--dist', dest='dist', type=str2bool, default=True,
                        help='distributed training or not')
    parser.add_argument('--launcher', default=None, type=str, help='Launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--backend', dest='backend', type=str, default='nccl',
                        help='backend for distributed training')
    parser.add_argument('--port', dest='port', required=True,
                        help='port of server')
    return parser.parse_args()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr
        optimizer.param_groups[2]['lr'] = lr * 10
        #optimizer.param_groups[3]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
            
def main():
    """Create the model and start the training."""
    global args
    args = get_arguments()
    if args.dist:
        init_dist(args.launcher, backend=args.backend)
    world_size = 1
    rank = 0
    if args.dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'Deeplab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from, strict=False)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        model.load_state_dict(saved_state_dict, strict=False)
    elif args.model == 'DeeplabVGGBN':
        deeplab_vggbn.BatchNorm = SyncBatchNorm2d
        model = deeplab_vggbn.DeeplabVGGBN(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
            model.load_state_dict(saved_state_dict, strict=False)
            del saved_state_dict
        
    model.train()
    model.to(device)
    if args.dist:
        broadcast_params(model)
    
    if rank == 0:
        print(model)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)

    model_D1.train()
    model_D1.to(device)
    if args.dist:
        broadcast_params(model_D1)
    if args.restore_D is not None:
        D_dict = torch.load(args.restore_D)
        model_D1.load_state_dict(D_dict, strict=False)
        del D_dict

    model_D2.train()
    model_D2.to(device)
    if args.dist:
        broadcast_params(model_D2)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    train_data = GTA5BDDDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    train_sampler = None
    if args.dist:
        train_sampler = DistributedSampler(train_data)
    trainloader = data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=False if train_sampler else True, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

    trainloader_iter = enumerate(cycle(trainloader))
    
    target_data = BDDDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set)
    target_sampler = None
    if args.dist:
        target_sampler = DistributedSampler(target_data)
    targetloader = data.DataLoader(target_data,
                                   batch_size=args.batch_size, shuffle=False if target_sampler else True, num_workers=args.num_workers,
                                   pin_memory=False, sampler=target_sampler)


    targetloader_iter = enumerate(cycle(targetloader))

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard and rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    torch.cuda.empty_cache()
    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, size, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)
            interp = nn.Upsample(size=(size[1], size[0]), mode='bilinear', align_corners=True)
            
            pred1 = model(images)
            pred1 = interp(pred1)

            loss_seg1 = seg_loss(pred1, labels)
            
            loss = loss_seg1
                

            # proper normalization
            loss = loss / args.iter_size / world_size
            loss.backward()
            loss_seg_value1 += loss_seg1.item() / args.iter_size

            _, batch = targetloader_iter.__next__()
            # train with target
            images, _, _ = batch
            images = images.to(device)

            pred_target1 = model(images)
            pred_target1 = interp_target(pred_target1)

            D_out1 = model_D1(F.softmax(pred_target1))
            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv_target1 * loss_adv_target1
            loss = loss / args.iter_size / world_size

            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            D_out1 = model_D1(F.softmax(pred1))
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))
            loss_D1 = loss_D1 / args.iter_size / 2 / world_size
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            # train with target
            pred_target1 = pred_target1.detach()
            D_out1 = model_D1(F.softmax(pred_target1))
            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))
            loss_D1 = loss_D1 / args.iter_size / 2 / world_size
            loss_D1.backward()
            if args.dist:
                average_gradients(model)
                average_gradients(model_D1)
                average_gradients(model_D2)

            loss_D_value1 += loss_D1.item()

        optimizer.step()
        optimizer_D1.step()
        
        if rank == 0:
            if args.tensorboard:
                scalar_info = {
                    'loss_seg1': loss_seg_value1,
                    'loss_seg2': loss_seg_value2,
                    'loss_adv_target1': loss_adv_target_value1,
                    'loss_adv_target2': loss_adv_target_value2,
                    'loss_D1': loss_D_value1 * world_size,
                    'loss_D2': loss_D_value2 * world_size,
                }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.snapshot_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

            if i_iter >= args.num_steps_stop - 1:
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
                torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
                break

            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
                torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
    print(args.snapshot_dir)
    if args.tensorboard and rank == 0:
        writer.close()


if __name__ == '__main__':
    main()
