from __future__ import print_function
import torch
import torch.nn.functional as F
from models.metrics import *
import time
import os
import numpy as np
from PIL import Image


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    ### (128.128.128)为颜色
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    shift = [(w-nw)//2, (h-nh)//2]
    return new_image, scale, shift


def set_params(params, network, weight_decay, lr):
    params_dict = dict(network.named_parameters())
    for key, value in params_dict.items():
        if key[-4:] == 'bias':
            params += [{'params': value, 'weight_decay': 0.0, 'lr': lr * 2}]
        else:
            params += [{'params': value, 'weight_decay': weight_decay, 'lr': lr}]
    return params


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('load success !')


def save_model(model, save_path, name, iter_cnt, use_multi_gpu):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    # multi gpu training
    if use_multi_gpu:
        torch.save(model.module.state_dict(), save_name)
    else:
        torch.save(model.state_dict(), save_name)

    return save_name


def save_checkpoint(model, classifier, optimizer, save_path, subtitle, use_multi_gpu):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_path = os.path.join(save_path, str(subtitle))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_name = os.path.join(checkpoint_path, 'model.pth')
    optimizer_name = os.path.join(checkpoint_path, 'optimizer.pth')
    classifier_name = os.path.join(checkpoint_path, 'classifier.pth')

    if use_multi_gpu:
        if model is not None:
            torch.save(model.module.state_dict(), model_name)
    else:
        if model is not None:
            torch.save(model.state_dict(), model_name)
    if classifier is not None:
        torch.save(classifier.state_dict(), classifier_name)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_name)


def draw_net_graph(model):
    import hiddenlayer as hl
    g = hl.build_graph(model, torch.zeros([1, 1, 128, 128]))
    g.save(os.path.join('./', "resnetface20.pdf"))


def moving_average(net1, net2, alpha=1.0):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



