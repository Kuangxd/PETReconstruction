import os
import sys
sys.path.insert(1, '/data/k003166/90-release/01-TICCGAN')


import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util.visualizer import save_images2
from util import html
import cv2
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
from scipy.ndimage import uniform_filter, gaussian_filter

def set_options():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.no_lsgan = 1
    opt.lr = 0.1
    opt.beta1 = 0.99

    opt.name = 'resnet_block6'
    opt.which_model_netG = 'resnet_6blocks'
    opt.phase = 'test'

    model = create_model(opt)

    opt.which_epoch = str('latest')
    model.setup(opt)
    return model

def dl_function(img_cv2):
    model = set_options()
    img = torch.from_numpy(img_cv2) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).cuda().float()

    model.set_input_pet(img)
    model.test()
    output = model.return_forward()

    output = output.squeeze().cpu().numpy().transpose((1, 2, 0))

    output[output<0] = 0.0
    output[output>1] = 1.0

    output = output * 255
    # pdb.set_trace()
    return output

def dl_function_grad(img_cv2, item1, item2):
    model = set_options()

    img = torch.from_numpy(img_cv2) / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).cuda().float()

    img.requires_grad = True

    model.set_input_pet(img)
    model.test_grad()
    output2 = model.return_forward()

    output = output2.clone()

    output[output<0] = 0.0
    output[output>1] = 1.0

    item1 = torch.from_numpy(item1) / 255.0
    item1 = item1.permute(2, 0, 1).unsqueeze(0).cuda().float()
    item2 = torch.from_numpy(item2) / 255.0
    item2 = item2.permute(2, 0, 1).unsqueeze(0).cuda().float()
    print((output-item1-item2).max().detach().cpu(), (output-item1-item2).min().detach().cpu())
    gradients = torch.autograd.grad(outputs=torch.pow(output-item1-item2, 2), inputs=img, grad_outputs=torch.ones_like(output), create_graph=False)

    return gradients[0].detach().squeeze().cpu().numpy().transpose((1, 2, 0))
