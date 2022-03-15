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

# load the labels text file
# labels = json.load(open("dependencies/labels_map.txt", "r"))

# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
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

def predict(output, topk=(1,)):
    '''
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

    '''
    output = torch.tensor(output)
    for idx in torch.topk(output, k=5).indices.squeeze(0).tolist():
        prob = torch.softmax(output, dim=1)[0, idx].item()
        # print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
        print('({label:<1},{p:4.1f}%)'.format(label=idx, p=prob*100), end = '')

    print('')


# read the image
fname = "/mnt/work/dataset/dlf/dlf_vca_v2.1_dhdcoco/val/car/00000000017188.png"
# fname = "/mnt/work/dataset/dlf/dlf_vca_v2.1_dhdcoco/val/motorcycle/00000000011437.png"
# fname = "/mnt/work/dataset_wd/DLFilter/debugging_dataset_tf/testing/clutter_background/extra5205.png"
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# pre-process the image like mobilenet and resize it to 224x224
img = pre_process_edgetpu(img, (224, 224, 3))
'''
plt.axis('off')
plt.imshow(img)
plt.show()
'''

# channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
img = np.array(img).transpose((2,0,1))     # transpose(2,0,1)    

# create a batch of 1 (that batch size is buned into the saved_model)
img_batch = np.expand_dims(img, axis=0)

# load model
sess = rt.InferenceSession("/mnt/work/model_pool/dl_filter/DLF-VCA-v2.1-EL/elite0-v2.1.onnx")

input_name = sess.get_inputs()[0].name
print('input_name=', input_name, 'img_batch.shape=', img_batch.shape)

# run inference
results = sess.run([], {input_name: img_batch})
results = results[0]
print('results=', results)

pred = predict(results, topk=(1, 5))
print('pred=', pred)

obj_labels = ['background','bag','bicycle','car','cyclist','motorcycle','person','truck','van']
print('obj_labels=', obj_labels)
print('fname=', fname)

