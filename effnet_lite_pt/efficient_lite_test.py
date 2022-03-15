import argparse
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import onnxruntime as rt
import cv2
import json

# to fix current path vs symbolic path issue
import site
import os
site.addsitedir(os.path.dirname(__file__))

import onnx
import geffnet

from PIL import Image
from torchvision.transforms import ToTensor
# from torch2trt import torch2trt

import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
# parser.add_argument('output', metavar='ONNX_FILE',
#                    help='output model filename')
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
parser.add_argument("--input_dir", type=str,
      default=f"/mnt/work/dataset/dlf/dlf_vca_v2.1_dhdcoco/val/car",
      help="coco image dir, default: %(default)s")

def predict(output, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.cpu().numpy().tolist()
    # print('pred=', pred)
    return pred

# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]

    # print('new_img.shape=', img.shape)
    return img

# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    print('img_w, img_h=(', width, height, ') ', 'resize-to (', w, h, ') ', 'crop-to (', out_height, out_width, ')')
    return img

# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

def resize_img(img, dims, inter_pol=cv2.INTER_LINEAR):
    output_height, output_width, _ = dims

    img = cv2.resize(img, (output_width, output_height), interpolation=inter_pol)
    return img

def batch_run(model, device, args):
    i = 0
    pytorch_time = []

    for path, dirs, files in os.walk(args.input_dir):
        for my_file in files:
            filename, ext = os.path.splitext(my_file)
            if ext == '.png' or ext == '.jpg':
                i = i + 1
                if i > 100:
                    break
                file_name = path + '/' + my_file
                print('file_name=', file_name)
                # https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4
                img = cv2.imread(file_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                '''
                plt.axis('off')
                plt.imshow(img)
                plt.show()
                '''
                # pre-process the image and resize it to 224x224
                pixels = 224
                batch = 1
                # method 1
                # img = pre_process_edgetpu(img, (pixels, pixels, 3))
                # method 2
                img = resize_img(img, (pixels, pixels, 3))
                '''
                plt.axis('off')
                plt.imshow(img)
                plt.show()
                '''

                # create a batch of 1 (that batch size is buned into the saved_model)
                # img_batch = np.expand_dims(img, axis=0)

                # image batch
                input = torch.from_numpy(img).view(batch,3,pixels,pixels).float().to(device)

                # Inference
                tic = time.time()
                out = model(input)
                toc = time.time()
                print('out=', out)

                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                pred = predict(out, topk=(1, 5))
                # print('pred=', pred, 'in=', file_name)
                flat_list = [item for sublist in pred for item in sublist]
                print('pred=', flat_list, 'in=', file_name)
                
                if i % 10 == 0:
                    print('Infer time(', i, ')=', (toc-tic)* 1000, ' ms')
                if i > 0 :
                    pytorch_time.append(toc-tic)

    # 8 image categories: 
    # object = ['0:person','1:cyclist','2:car','3:van','4:truck','5:motorcycle','6:bicycle','7:bag']
    # Alphabetic order
    object = ['0:bag','1:bicycle','2:car','3:cyclist','4:motorcycle','5:person','6:truck','7:van']

    print("batch_run finished")
    print('object=', object)

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

    batch_run(model, device, args)

    '''
    pytorch_time = []
    batch = args.batch_size  # 16
    for x in range(100):
    
        img = np.random.rand(batch,pixels,pixels,3)
        input = torch.from_numpy(img).view(batch,3,pixels,pixels).float().to(device)
        tic = time.time()
        out = model(input)
        toc = time.time()
        print('out=', out)

        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        predict(out, topk=(1, 5))

        if x % 10 == 0:
            print('Infer time(', x, ')=', (toc-tic)* 1000, ' ms')
        if x > 0 :
            pytorch_time.append(toc-tic)
    print("pytorch inference took: ", np.mean(np.asarray(pytorch_time))*1000, ' ms')
    print("pytorch FPS is: ", 1/np.mean(np.asarray(pytorch_time))*batch)
    print('batch = ', batch)
    '''



if __name__ == '__main__':
    main()


