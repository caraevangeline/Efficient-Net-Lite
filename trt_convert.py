import argparse
import torch
import numpy as np

# to fix current path vs symbolic path issue
import site
import os
site.addsitedir(os.path.dirname(__file__))

import onnx
import geffnet

from PIL import Image
from torchvision.transforms import ToTensor
from torch2trt import torch2trt

import time


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='ONNX_FILE',
                    help='output model filename')
parser.add_argument('--model', '-m', metavar='MODEL', default='mobilenetv3_large_100',
                    help='model architecture (default: mobilenetv3_large_100)')
parser.add_argument('--opset', type=int, default=10,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')


def main():
    args = parser.parse_args()
    print(args)

    args.pretrained = True
    if args.checkpoint:
        args.pretrained = False

    print("==> Creating PyTorch {} model".format(args.model))
    # NOTE exportable=True flag disables autofn/jit scripted activations and uses Conv2dSameExport layers
    # for models using SAME padding
    model = geffnet.create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        exportable=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on device ' + str(device))

    # load the image
    pixels = 224
    img = np.random.rand(pixels,pixels,3)
    input = torch.from_numpy(img).view(1,3,pixels,pixels).float().to(device)
    print("input size is..", input.shape)

    # set to eval and send to gpu
    model = model.eval().to(device)
    print("model set!")

    pytorch_time = []
    batch = args.batch_size  # 16
    for x in range(100):
    
        img = np.random.rand(batch,pixels,pixels,3)
        input = torch.from_numpy(img).view(batch,3,pixels,pixels).float().to(device)
        tic = time.time()
        out = model(input)
        toc = time.time()
        if x % 10 == 0:
            print('Infer time(', x, ')=', (toc-tic)* 1000, ' ms')
        if x > 0 :
            pytorch_time.append(toc-tic)
    print("pytorch inference took: ", np.mean(np.asarray(pytorch_time))*1000, ' ms')
    print("pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)
    print('batch = ', batch)
    
    # try Pytorch FP16 inference
    # ToDo

    # export the model
    input_names = [ "input_0" ]
    output_names = [ "output_0" ]

    print('exporting model to trt...')
    tic = time.time()
    model_trt = torch2trt(model, [input], max_batch_size=batch)
    toc = time.time()
    print("conversion completed! took:", toc-tic)

    trt_time = []
    for x in range(100):
    
        img = np.random.rand(batch,pixels,pixels,3)
        input = torch.from_numpy(img).view(batch,3,pixels,pixels).half().to(device)
        tic = time.time()
        out = model_trt(input)
        toc = time.time()
        if x % 10 == 0:
            print('Infer time(', x, ')=', (toc-tic)* 1000, ' ms')
        if x > 0 :
            trt_time.append(toc-tic)
    print("trt inference took: ", np.mean(np.asarray(trt_time))*1000, ' ms')
    print("trt FPS is: ", 1/np.mean(np.asarray(trt_time))*batch)


if __name__ == '__main__':
    main()


'''
# starting point: 
# https://github.com/kentaroy47/pytorch-onnx-tensorrt-CIFAR10.git

import argparse
import torch

from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from torch2trt import torch2trt

import time

# exporter settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='res18', help="set model checkpoint path")
parser.add_argument('--model_out', type=str, default='resnet18.onnx')
#parser.add_argument('--image', type=str, required=True, help='input image to use')

args = parser.parse_args() 
print(args)

# exporter settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='res18', help="set model checkpoint path")
parser.add_argument('--model_out', type=str, default='resnet18.onnx')
#parser.add_argument('--image', type=str, required=True, help='input image to use')

args = parser.parse_args() 
print(args)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))


# load the image
pixels = 32
img = np.random.rand(pixels,pixels,3)
input = torch.from_numpy(img).view(1,3,pixels,pixels).float().to(device)
print("input size is..", input.shape)

# load the model
from models import *
model = ResNet18()
checkpoint = torch.load(args.model)

# model.load_state_dict(checkpoint['net'])
# LLJ
model.load_state_dict(checkpoint['net'], strict=False)

# set to eval and send to gpu
model = model.eval().to(device)
print("model set!")

pytorch_time = []
batch = 32
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).float().to(device)
    tic = time.time()
    out = model(input)
    toc = time.time()
    pytorch_time.append(toc-tic)
print("pytorch inference took: ", np.mean(np.asarray(pytorch_time)))
print("pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)

# try Pytorch FP16 inference
from fp16util import network_to_half
model2 = network_to_half(model)
pytorch_time = []
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).half().to(device)
    tic = time.time()
    out = model2(input)
    toc = time.time()
    pytorch_time.append(toc-tic)
print("FP16 pytorch inference took: ", np.mean(np.asarray(pytorch_time)))
print("FP16 pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)

del model2

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to trt...')
tic = time.time()
model_trt = torch2trt(model, [input], max_batch_size=batch)
toc = time.time()
print("conversion completed! took:", toc-tic)

trt_time = []
for x in range(100):
    
    img = np.random.rand(batch,pixels,pixels,3)
    input = torch.from_numpy(img).view(batch,3,pixels,pixels).half().to(device)
    tic = time.time()
    out = model_trt(input)
    toc = time.time()
    trt_time.append(toc-tic)
print("trt inference took: ", np.mean(np.asarray(trt_time)))
print("trt FPS is: ", 1/np.mean(np.asarray(trt_time))*batch)

'''