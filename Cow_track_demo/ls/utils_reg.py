"""
common functions are saved in this script
"""
from PIL import Image
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import re
from torchvision import transforms as T
import os
import pandas as pd

from ls.seresnet import se_resnet50
from ls.networks import resnetface20
from ls.metrics import SVAMProduct


def get_model_keywords(dataset:str):
    keywords = ['xz', 'mask125', 'sameway', 'rot', '528']
    datasetInfo = [key.capitalize() for key in keywords if key in dataset.lower()]
    model_keywords = ''.join(datasetInfo)
    return model_keywords

def get_subtitle(config: dict, transform: T.Compose):
    re_transform = re.findall(r"degrees=\(-*\d+, \d+\)", str(transform))
    degrees = re.findall(r"-*\d+", str(re_transform))
    data_aug = 'rot' + str(int(degrees[1])) if len(degrees) == 2 else 'rotX'
    if str('RandomHorizontalFlip') in str(transform):
        data_aug = data_aug + 'Hor'
    if str('RandomVerticalFlip') in str(transform):
        data_aug = data_aug + 'Ver'

    finetune, triplet = '', ''
    if config['phase']=='finetune':
        finetune, triplet = '_finetune',''
    if config['phase']=='triplet':
        finetune, triplet = '_finetune','_triplet'

    dataset = config['train_root'].split('/')[-1]
    keywords = ['xz', 'mask125', 'sameway', 'rot', '528', 'blur', 'equalize', 'noise']
    datasetInfo = [key.capitalize() for key in keywords if key in dataset.lower()]
    datasetInfo = ''.join(datasetInfo)

    subtitle = "{dataset}_{backbone}_{data_aug}_{shape}{finetune}{triplet}".format(
        dataset=datasetInfo,
        backbone=config['backbone'],
        data_aug=data_aug,
        shape=str(config['input_shape'][2]),
        finetune=finetune,
        triplet=triplet)
    return subtitle


def load_network_structure(config: dict, single_model=False, num_classes=1000):
    """
    load network structure
    :param config:
    :param single_model: weather return network only
    :param num_classes:
    :return:
    """
    global backbone, classifier

    if config['backbone'] == 'resnetface20':
        backbone = resnetface20(input_shape=config['input_shape'])
    elif config['backbone'] == 'se_resnet50':
        backbone = se_resnet50(pretrained='imagenet')
        if config['input_shape'][1] == 128:
            # 128 bodai
            backbone.avg_pool = nn.AvgPool2d(4, stride=1)
        elif config['input_shape'][1] == 320:
            # 320 cow body
            backbone.avg_pool = nn.AvgPool2d(10, stride=1)
            # 192 cow body register
        elif config['input_shape'][1] == 192:
            backbone.avg_pool = nn.AvgPool2d(6, stride=1)
        backbone.last_linear = nn.Linear(2048, 512)
    elif 'efficientnet' in config['backbone']:
        backbone = EfficientNet.from_pretrained(config['backbone'])
        feature = backbone._fc.in_features
        backbone._fc = nn.Linear(in_features=feature, out_features=512, bias=True)
    else:
        assert 'backbone error'

    if single_model:
        return backbone

    if config['phase']=='train':
        classifier = torch.nn.Linear(512, num_classes, bias=False)
    elif config['phase']=='finetune':
        classifier = SVAMProduct(512, num_classes, s=30, m=0.35, t=1.2)
    elif config['phase']=='triplet':
        classifier = SVAMProduct(512, num_classes, s=30, m=0.35, t=1.2)
    else:
        assert 'config->phase error'
        exit()

    criterion = torch.nn.CrossEntropyLoss()

    return backbone, classifier, criterion


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    shift = [(w - nw) // 2, (h - nh) // 2]
    return new_image, scale, shift


# def transform_(new_img):
#     img = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
#     mean = (0.5, 0.5, 0.5)
#     std = (0.5, 0.5, 0.5)
#     normalized_img = (img / 255. - mean) / std
#     return normalized_img


def prepare_image(img_path, dst_size, transform):
    try:
        img = Image.open(img_path)
    except IOError:
        raise Exception("Error: read %s fail" % img_path)
    new_img, _, _ = letterbox_image(img, dst_size)
    input_data = transform(new_img)
    return input_data


def get_dirs_all_data(imgdir):
    """
    获取imgdir下所有label内的图片, 返回所有图片的路径
    """
    img_list = []
    for each in os.listdir(imgdir):
        sub_dir = os.path.join(imgdir, each)
        files = os.listdir(sub_dir)
        paths = []
        for each_file in files:
            # 去除 'Thumbs.db' 文件
            if 'Thumbs.db' in each_file: continue
            if '副本' in each_file: continue
            paths.append(os.path.join(sub_dir, each_file))
        img_list.extend(paths)
    return img_list

def save_to_final_result(config, acc_data, model_path):
    data = {
        # max_th1_tp_rate： 需要修改至after_filtered_by_th1
        # max_th_e_1： 修改至 fpr_th_e_1
        "train_dataset":[config['train_root']],
        "backbone":[config['backbone']],
        "data_aug":[config['checkpoint_subtitle']],
        "input_shape":[config['input_shape'][2]],
        "phase":[config['phase']],
        "model_path":[model_path],
        "ref_dataset":[config['ref_dir']],
        "test_dataset":[config['test_dir']],
        "max_acc":[acc_data[0]],
        "max_auc":[acc_data[1]],
        "max_th_e_1":[acc_data[2]],
        "max_th1_acc":[acc_data[3]],
        "max_num1":[acc_data[4]],
        "max_th_e_2":[acc_data[5]],
        "max_th2_acc":[acc_data[6]],
        "max_num2":[acc_data[7]],
        "max_th_e_3":[acc_data[8]],
        "max_th3_acc":[acc_data[9]],
        "max_num3":[acc_data[10]],
        "total_num":[acc_data[11]],
    }
    df_data = pd.DataFrame(data)
    if os.path.exists(os.path.join(config['result_file'])):
        df_data.to_csv(os.path.join(config['result_file']), mode='a', header=False, index=False)
    else:
        df_data.to_csv(os.path.join(config['result_file']), mode='w', header=True, index=False)