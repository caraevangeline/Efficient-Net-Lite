""" ONNX export script

"""
import argparse
import torch
import numpy as np
import torch.nn as nn
import cv2

# bugfix: current path vs symbolic path issue
import site
import os
site.addsitedir(os.path.dirname(__file__))

import onnx
# import geffnet
from efficientnet_lite import efficientnet_lite_params, build_efficientnet_lite
CROP_PADDING = 32
MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='ONNX_FILE',
                    help='output model filename')
parser.add_argument('--ptfile', default='test.pt',
                    help='output pt model')
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

def load_checkpoint(net, checkpoint, use_gpu):
    from collections import OrderedDict

    temp = OrderedDict()
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    use_gpu_s = False
    if use_gpu_s:
        for k in checkpoint:
            k2 = 'module.'+k if not k.startswith('module.') else k
            temp[k2] = checkpoint[k]
    else:
        # bugfix: error(s) in loading state_dict for efficientdet on CPU inference
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            temp[name] = v

    net.load_state_dict(temp, strict=True)

def main():
    args = parser.parse_args()

    input_size = efficientnet_lite_params[args.model][2]
    args.input_size = input_size
    print('input_size=', input_size)

    use_gpu = False
    if torch.cuda.is_available():
       use_gpu = True

    print('args.model, args.num_classes=', args.model, args.num_classes)

    model = build_efficientnet_lite(args.model, args.num_classes)

    if use_gpu:
        # model = nn.DataParallel(model)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=None if use_gpu else 'cpu')

    load_checkpoint(model, checkpoint, use_gpu)

    model.eval()

    example_input = torch.randn((args.batch_size, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)

    example_input = example_input.to(device)
    model(example_input)
    # model(img_batch)

    # for libtorch: to train a model on GPU and do inference on CPU
    print("==> Exporting pt model for libtorch at '{}'".format(args.ptfile))
    model_cpu = build_efficientnet_lite(args.model, args.num_classes)
    device_cpu = torch.device("cpu")
    model_cpu = model_cpu.to(device_cpu)
    checkpoint_cpu = torch.load(args.checkpoint, map_location='cpu')
    load_checkpoint(model_cpu, checkpoint_cpu, False)
    model_cpu.eval()
    example_cpu = torch.randn((args.batch_size, 3, args.img_size or 224, args.img_size or 224), requires_grad=True)
    example_cpu = example_cpu.to(device_cpu)
    model_cpu(example_cpu)

    traced_script_module = torch.jit.trace(model_cpu, example_cpu)
    traced_script_module.save(args.ptfile)

    print("==> Exporting model to ONNX format at '{}'".format(args.output))
    input_names = ["input0"]
    output_names = ["output0"]
    dynamic_axes = {'input0': {0: 'batch'}, 'output0': {0: 'batch'}}
    if args.dynamic_size:
        dynamic_axes['input0'][2] = 'height'
        dynamic_axes['input0'][3] = 'width'
    if args.aten_fallback:
        export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    else:
        export_type = torch.onnx.OperatorExportTypes.ONNX

    torch_out = torch.onnx._export(
        model, example_input, args.output, export_params=True, verbose=True, input_names=input_names,
        output_names=output_names, keep_initializers_as_inputs=args.keep_init, dynamic_axes=dynamic_axes,
        opset_version=args.opset, operator_export_type=export_type)

    print("==> Loading and checking exported model from '{}'".format(args.output))
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)  # assuming throw on error
    print("==> Passed")

    if args.keep_init and args.aten_fallback:
        import caffe2.python.onnx.backend as onnx_caffe2
        # Caffe2 loading only works properly in newer PyTorch/ONNX combos when
        # keep_initializers_as_inputs and aten_fallback are set to True.
        print("==> Loading model into Caffe2 backend and comparing forward pass.".format(args.output))
        caffe2_backend = onnx_caffe2.prepare(onnx_model)
        B = {onnx_model.graph.input[0].name: x.data.numpy()}
        c2_out = caffe2_backend.run(B)[0]
        np.testing.assert_almost_equal(torch_out.data.numpy(), c2_out, decimal=5)
        print("==> Passed")
   
if __name__ == '__main__':
    main()