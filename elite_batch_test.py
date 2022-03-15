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
import onnxruntime as rt

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
    # /mnt/work/model_pool/dl_filter/DLF-VCA-v2.1-EL/elite0-v2.1.onnx
    parser.add_argument('--eval_onnx', type=str, default='./elite0-v2.1.onnx', help='path for eval model')

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
    parser.add_argument('--use_cpu', type=bool, default=False, help='run on CPU')
    parser.add_argument('--display_interval', type=int, default=20, help='display interval')
    parser.add_argument('--val_interval', type=int, default=10000, help='val interval')
    parser.add_argument('--save_interval', type=int, default=10000, help='save interval')

    parser.add_argument('--train_dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/val', help='path to validation dataset')
    parser.add_argument('--image_total', type=int, default=10, help='save interval')

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
    args.input_size = input_size
    print('input_size=', input_size)

    use_gpu = False
    if torch.cuda.is_available():
       use_gpu = True
    # LLJ
    if args.use_cpu:
        use_gpu = False
    print('use_gpu=', use_gpu, 'args.use_cpu=', args.use_cpu)

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

    preprocess = transforms.Compose([
            transforms.Resize(input_size + CROP_PADDING, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_RGB, STDDEV_RGB)
        ])
    args.preprocess = preprocess
    
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
            load_checkpoint(model, checkpoint, use_gpu)
            validate(model, device, args)
        exit(0)

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    # load model: args.eval_onnx
    # sess = rt.InferenceSession("/mnt/work/model_pool/dl_filter/DLF-VCA-v2.1-EL/elite0-v2.1.onnx")
    sess = rt.InferenceSession(args.eval_onnx)
    input_name = sess.get_inputs()[0].name
    print('input_name=', input_name, 'args.eval_onnx=', args.eval_onnx)

    # https://github.com/ml-illustrated/EfficientNet-Lite-PyTorch
    # labels_map = json.load(open('labels_map.txt'))
    # labels_map = [labels_map[str(i)] for i in range(1000)]

    val_dataprovider = args.val_dataprovider
    preprocess = args.preprocess

    model.eval()

    # obj_labels = ['person','cyclist','car','van','truck','motorcycle','bicycle','bag']

    # Alphabetic order: dlf_vca_v2.1_dhdcoco dataset
    # obj_labels = ['bag','bicycle','car','cyclist','motorcycle','person','truck','van']

    # Alphabetic order: dlf_vca_v2.2 dataset
    obj_labels = ['background','bag','bicycle','car','cyclist','motorcycle','person','truck','van']


    i = 0
    pytorch_time = []
    predict_count = 0

    for path, dirs, files in os.walk(args.val_dir):
        files.sort()

        for my_file in files:
            filename, ext = os.path.splitext(my_file)
            if ext == '.png' or ext == '.jpg':
                i = i + 1
                if i > args.image_total:
                    break
                file_name = path + '/' + my_file
                # print('file_name=', file_name)
                # img = cv2.imread(file_name)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.open(file_name)
                '''
                plt.axis('off')
                plt.imshow(img)
                plt.show()
                '''
                obj_idx = 100
                for count, label in enumerate(obj_labels):
                    if label in file_name:
                        obj_idx = count

                    

                img = preprocess(img)
                # create a batch of 1 (that batch size is buned into the saved_model)
                # input = img.to(device)
                img_batch = np.expand_dims(img, axis=0)
                pixels = args.input_size
                input = torch.from_numpy(img_batch).view(1,3,pixels,pixels).float().to(device)

                # Inference
                tic = time.time()
                out = model(input)
                toc = time.time()

                fmt_i = "{0:3}".format(i)
                fmt_tick = "{:.2f}".format((toc-tic)* 1000)

                # '''
                # print('out=', out)

                # prec1, prec5 = accuracy(output, target, topk=(1, 5))
                pred = predict(out, topk=(1, 5))
                # print('pred=', pred, 'in=', file_name)
                flat_list = [item for sublist in pred for item in sublist]

                for item in pred[0]: 
                    print('item=', item)
                    if item == 0:
                        predict_count += 1
                print('idx=', obj_idx, 'pred=', flat_list, 'in=', file_name, 'count=', fmt_i, '/', args.image_total, fmt_tick, 'ms', predict_count)
                # '''
                
                # if i % 10 == 0:
                #    print('Infer time(', i, ')=', (toc-tic)* 1000, ' ms')
                if i > 0 :
                    pytorch_time.append(toc-tic)

                # ONNX model run inference
                results = sess.run([], {input_name: img_batch})
                results = results[0]
                '''
                print('results=', results)
                print('pytorch_model_out=', out)
                print('img_batch.shape=', img_batch.shape)
                print('input.shape=', input.shape)
                '''

    
    print("batch_run finished")

    print('object label=', end = '')
    for count, value in enumerate(obj_labels):
        print(count, value, end = '   ')
    print('Finished')

def predict(output, topk=(1,)):
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.cpu().numpy().tolist()
    # print('pred=', pred)

    for idx in torch.topk(output, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(output, dim=1)[0, idx].item()
        # print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
        print('({label:<1},{p:4.1f}%)'.format(label=idx, p=prob*100), end = '')

    return pred

def load_checkpoint(net, checkpoint, use_gpu):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    if use_gpu:
        for k in checkpoint:
            k2 = 'module.'+k if not k.startswith('module.') else k
            temp[k2] = checkpoint[k]
    else:
        # bugfix: error(s) in loading state_dict for efficientdet on CPU inference
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            temp[name] = v

    net.load_state_dict(temp, strict=True)

if __name__ == "__main__":
    main()
