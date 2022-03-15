import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
import time
import logging
import argparse

# to fix current path vs symbolic path issue
import site
import os
site.addsitedir(os.path.dirname(__file__))

from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
from utils.train_utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters

CROP_PADDING = 32
MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='efficientnet_lite0', help='name of model: efficientnet_lite0, 1, 2, 3, 4')

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str, default='./efficientnet_lite0.pth', help='path for eval model')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto_continue', type=bool, default=True, help='auto continue')
    parser.add_argument('--display_interval', type=int, default=20, help='display interval')
    parser.add_argument('--val_interval', type=int, default=10000, help='val interval')
    parser.add_argument('--save_interval', type=int, default=10000, help='save interval')

    parser.add_argument('--train_dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    input_size = efficientnet_lite_params[args.model_name][2]
    print('input_size=', input_size)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    assert os.path.exists(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            transforms.Resize(input_size + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_RGB, STDDEV_RGB)
        ])),
#        batch_size=200, 
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')
    print('args.model_name, args.num_classes=', args.model_name, args.num_classes)

    model = build_efficientnet_lite(args.model_name, args.num_classes)

    if use_gpu:
        model = nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            load_checkpoint(model, checkpoint)
            validate(model, device, args)
        exit(0)

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    topk_classes = min(5, args.num_classes)

    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250000000
    t1  = time.time()
    i = 0
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            i = i + 1
            if i > 2:
                print("early exit, i =", i)
                break

            data, target = val_dataprovider.next()
            print("data=", data)
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            output = model(data)
            print('data.shape=', data.shape)

            # LLJ: test
            # print('i=', i,'shape=', output.shape, 'output=', output) 
            print('i=', i,'out=', output) 

            prec1, prec5 = accuracy(output, target, topk=(1, topk_classes))

            # LLJ: test
            print('target=', target.cpu().numpy().tolist()) 
            # print('target=', target) 

            n = data.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    # object = ['0:person','1:cyclist','2:car','3:van','4:truck','5:motorcycle','6:bicycle','7:bag']
    # Alphabetic order
    object = ['0:bag','1:bicycle','2:car','3:cyclist','4:motorcycle','5:person','6:truck','7:van']

    print("batch_run finished")
    print('object=', object)

def load_checkpoint(net, checkpoint):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        k2 = 'module.'+k if not k.startswith('module.') else k
        temp[k2] = checkpoint[k]

    net.load_state_dict(temp, strict=True)

if __name__ == "__main__":
    main()
