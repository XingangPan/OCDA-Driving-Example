import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
import model.deeplab_vggbn as deeplab_vggbn
from model.syncbn_layer import SyncBatchNorm2d
from dataset.bdd_dataset import BDDDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './C-Driving/val/compound'
DATA_LIST_PATH = './dataset/bdd_list/val/3domains.txt'
SAVE_PATH = './result/bdd100k'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_confidence(output):
    softmax = nn.Softmax(dim=1)
    prob = softmax(output) # 1 c h w
    prob, _ = torch.max(prob, dim=1) # 1 h w
    return torch.mean(prob)

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
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--save-confidence", action='store_true', help="save confidence.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG
    elif args.model == 'DeeplabVGGBN':
        deeplab_vggbn.BatchNorm = SyncBatchNorm2d
        model = deeplab_vggbn.DeeplabVGGBN(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict, strict=False)
    
    print(model)

    device = torch.device("cuda" if not args.cpu else "cpu")
    model = model.to(device)

    model.eval()

    testloader = data.DataLoader(BDDDataSet(args.data_dir, args.data_list, crop_size=(960, 540), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)    # 960 540

    interp = nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=True)
    
    if args.save_confidence:
        select = open('list.txt', 'w')
        c_list = []

    for index, batch in enumerate(testloader):
        if index % 10 == 0:
            print('%d processd' % index)
        image, _, name = batch
        image = image.to(device)

        output = model(image)

        if args.save_confidence:
            confidence = get_confidence(output)
            confidence = confidence.cpu().item()
            c_list.append([confidence, name])
            name = name[0].split('/')[-1]
            save_path = '%s/%s_c.txt' % (args.save, name.split('.')[0])
            record = open(save_path, 'w')
            record.write('%.5f' % confidence)
            record.close()
        else:
            name = name[0].split('/')[-1]
            
        output = interp(output).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        output.save('%s/%s' % (args.save, name[:-4] + '.png'))
        output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))
    
    def takeFirst(elem):
        return elem[0]
    
    if args.save_confidence:
        c_list.sort(key=takeFirst, reverse=True)
        length = len(c_list)
        for i in range(length // 3):
            print(c_list[i][0])
            print(c_list[i][1])
            select.write(c_list[i][1][0])
            select.write('\n')
        select.close()
        
    print(args.save)


if __name__ == '__main__':
    main()
