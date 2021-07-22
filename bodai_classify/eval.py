from __future__ import print_function
import copy
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from models.networks import resnetface20, resnetface36
from utils import letterbox_image
import yaml
from sklearn.metrics import roc_curve, auc
import time
import matplotlib.pyplot as plt
import PIL
from models.seresnet import se_resnet50
import torch.nn as nn
import cv2
from dataset.dataset import BoDaiDataset
from torch.utils import data
# 打印各个类别的精确率和召回率
### print every class precision and recall
from sklearn.metrics import classification_report
#
# y_true = [0, 1, 2, 2, 2]
# y_pred = [0, 0, 2, 2, 1]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))
from pandas import DataFrame


def prepare_image(img_path, dst_size, transform):
    try:
        img = Image.open(img_path)
    except IOError:
        raise Exception("Error: read %s fail" % img_path)

    new_img, _, _ = letterbox_image(img, dst_size)
    # add
    # new_img = PIL.ImageOps.equalize(new_img, mask=None)
    input_data = transform(new_img)
    return input_data


def infer_single_image():
    pass


def infer_batch(model, batch):
    pass


### just print info
@torch.no_grad()
def infer_and_show(model, test_list, dst_size, label_name, is_show=True):
    # df = DataFrame(columns=['pic','ground_truth','predict'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_dict = {}
    for label in label_name:
        acc_dict[label] = [0, 0]  # all_pic_num, error_num

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean, std)
                           ])

    for one_test in test_list:
        image_show = cv2.imread(one_test)
        GT_label = one_test.split('/')[-2]

        input_data = prepare_image(one_test, dst_size, transform)
        input_data = input_data.unsqueeze(0).to(device)
        output = model(input_data)
        output = F.softmax(output, dim=1)
        index = output.cpu().data.numpy().argmax()
        label = label_name[index]
        print(20*"==")
        print(label,)
        print(output.cpu().data.numpy()[0][index],)
        # 统计该label下的检测图片的总数以及识别错误的总数
        acc_dict[GT_label][0] += 1  # 总数+1
        if GT_label == label:
            print("True")
        else:
            print("img dir",one_test)
            print("ground truth:", GT_label, 'predict:', label)
            print("False")
            acc_dict[GT_label][1] += 1

        if is_show:
            cv2.imshow("show", image_show)
            cv2.waitKey(0)
    # 打印各个类别的识别准确率
    for key, value in acc_dict.items():
        print(20 * "==")
        print("当前名称为: ", key)
        print("总数为: ", value[0])
        print("预测错误的个数为: ", value[1])
        print("准确率为: ", (value[0] - value[1]) / value[0])



@torch.no_grad()
def test(model, test_list, dst_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    images_list = []
    for line in test_list:
        if line[0] not in images_list: images_list.append(line[0])
        if line[1] not in images_list: images_list.append(line[1])

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean, std)])

    batch_size = 256
    feature_dict = {}
    cnt = 0
    batch_cnt = 0
    batch = None
    batch_keys = []

    t1 = time.time()
    while (cnt < len(images_list)):

        image_path = images_list[cnt]
        input_data = prepare_image(image_path, dst_size, transform)
        input_data = input_data.unsqueeze(0).to(device)
        batch_keys.append(image_path)
        if batch is None:
            batch = input_data
        else:
            batch = torch.cat([batch, input_data])

        cnt += 1
        batch_cnt += 1

        if batch_cnt == batch_size or cnt == len(images_list):
            t2 = time.time()
            # print(batch.shape)
            features = model(batch)
            # features_norm = F.normalize(features)
            print('batch infer time: %f' % (time.time() - t2))

            for j, feature_key in enumerate(batch_keys):
                # feature_dict[feature_key] = features_norm[j]
                feature_dict[feature_key] = features[j]

            batch_cnt = 0
            batch = None
            batch_keys = []
    print('feature infer time: %f' % (time.time() - t1))

    t1 = time.time()
    predicts = []
    labels = []
    feature_list1 = []
    feature_list2 = []
    for line in test_list:
        # feature1 = feature_dict[line[0]]
        # feature2 = feature_dict[line[1]]
        # predicts.append(torch.dot(feature1, feature2).cpu().numpy())
        # predicts.append(torch.cosine_similarity(feature1, feature2).cpu().numpy())
        feature_list1.append(feature_dict[line[0]])
        feature_list2.append(feature_dict[line[1]])
        labels.append(line[2])

    features1 = torch.stack(feature_list1)
    features2 = torch.stack(feature_list2)
    predicts = torch.cosine_similarity(features1, features2).cpu().numpy()

    print('sim calculate time: %f' % (time.time() - t1))

    return predicts, labels


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
            paths.append(os.path.join(sub_dir, each_file))
        img_list.extend(paths)
    return img_list


@torch.no_grad()
def infer_from_dataload(model, test_list, dst_size, label_name, save_csv,bodai_res_csv, is_show=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_rec_dict = {}  # 准确率, 召回率的字典
    errdf = DataFrame(columns=['pic', 'ground_truth',
                            'predict1', 'prob1',
                            'predict2', 'prob2',
                            'predict3', 'prob3' ])
    y_true = []  #
    y_pred = []  #

    for label in label_name:
        # (该label真实有多少个数, 预测成该label有多少个数, 预测成该类别且正确的个数)
        acc_rec_dict[label] = [0, 0, 0]  # all_gt_num, predict_num, predict_and_true

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean, std)
                           ])

    for one_test in test_list:
        # print("one_test.split()[1]", one_test)
        # exit(0)
        gt_label_index = int(one_test.split()[1])


        one_test = one_test.split()[0]
        # show img
        image_show = cv2.imread(one_test)
        GT_label = one_test.split('/')[-2]

        input_data = prepare_image(one_test, dst_size, transform)
        input_data = input_data.unsqueeze(0).to(device)
        output = model(input_data)
        output = F.softmax(output, dim=1)
        # index = output.cpu().data.numpy().argmax()
        opt = output.cpu().numpy()[0]
        # print("softmax:", opt)
        top_k_idx = opt.argsort()[::-1][0:3]
        index = output.cpu().data.numpy().argmax()
        label = label_name[index]
        print(20 * "==")
        print(label)
        print(output.cpu().data.numpy()[0][index])
        # 统计该label下的检测图片的总数以及识别错误的总数
        # (该label真实有多少个数, 预测成该label有多少个数, 预测成该类别且正确的个数)
        acc_rec_dict[GT_label][0] += 1  # 实际该类别的总数+1

        acc_rec_dict[label][1] += 1  # 实际预测的类别的总数+1

        if GT_label == label:
            print("True")
            acc_rec_dict[GT_label][2] += 1  # 预测成该类别且正确的个数+1
        else:
            print("False")
            # pred = []
            # for idx in top_k_idx:
            #     pred.append([label_name[idx], opt[idx]])
            errdf = errdf.append([{"pic": one_test, "ground_truth": GT_label,
                             "predict1": label_name[top_k_idx[0]], 'prob1': round(opt[top_k_idx[0]],2),
                             "predict2": label_name[top_k_idx[1]], 'prob2': round(opt[top_k_idx[1]],2),
                             # "predict3": label_name[top_k_idx[2]], 'prob3': round(opt[top_k_idx[2]],2),
                             }], ignore_index=True)
            # print("real", GT_label, "test", label)
            # 识别错误+1
            # acc_dict[GT_label][1] += 1

        if is_show:
            cv2.imshow("show", image_show)
            cv2.waitKey(0)
        # 统计真实和预测的
        y_true.append(gt_label_index)
        y_pred.append(index)

    # 打印各个类别的识别准确率
    acc_list = []
    rec_list = []
    for key, value in acc_rec_dict.items():
        print(20 * "==")
        print("当前名称为: ", key)
        print("总数为: ", value[0])
        print("召回率为: ", value[2], value[0])
        print("精确率为: ", value[2], value[1])
        if value[1]!=0:
            acc_list.append(value[2] / value[1])
            rec_list.append(value[2] / value[0])
        else:
            acc_list.append(0)
            rec_list.append(value[2] / value[0])
    print(20*"##")
    print("总的平均召回率为: ", np.mean(rec_list))
    print("总的平均精确率为: ", np.mean(acc_list))

    print(20 * "**")

    report = classification_report(y_true, y_pred, target_names=label_name, output_dict=True)
    print(report)
    # 以下的方式也可以
    # list size(0)# labels = [i for i in range(len(label_name))]
    # print(classification_report(y_true, y_pred, labels=labels))
    # 输出到csv文件
    import pandas as pd
    df = pd.DataFrame(report).transpose()
    # print(errdf)
    df.to_csv(os.path.join('output', bodai_res_csv), index=True, encoding ='GBK')
    errdf.to_csv(os.path.join('output', save_csv), index=True, encoding ='GBK')


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        ### 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
        ### 进而实现网络的加速。适用场景是网络结构固定
        torch.backends.cudnn.benchmark = True

    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    print(config)

    model_path = config['model_path']
    input_shape = config['input_shape']
    label_class = config['class_label']
    with open(label_class, 'r') as fd:
        lines = fd.readlines()
        label_name = [line.strip() for line in lines]
    num_class = len(label_name)
    print("num_class", num_class)
    model = se_resnet50(pretrained='imagenet')
    # 128
    # model.avg_pool = nn.AvgPool2d(4, stride=1)
    if config['input_shape'][1] == 128:
        # 128 bodai
        model.avg_pool = nn.AvgPool2d(4, stride=1)
        print("add")
    elif config['input_shape'][1] == 320:
        # 320 cow body
        model.avg_pool = nn.AvgPool2d(10, stride=1)

    model.last_linear = nn.Linear(2048, num_class)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    val_dir = config['val_root']

    val_list = get_dirs_all_data(val_dir)
    print("len(test_list)", len(val_list))
    # 从测试图片路径里, 统计每类的预测准确率
    # infer_and_show(model, val_list, (input_shape[2], input_shape[1]), label_name, False)

    # 从dataloder里, 统计每类的预测准确率
    val_dataset = BoDaiDataset(val_dir, label_class, phase='val', input_shape=[3, 128, 128])
    valloader = data.DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                # num_workers=cpu_count(),
                                num_workers=8,
                                pin_memory=True)
    val_list = [each for each in val_dataset.data_list]

    # infer_and_show(model, val_list, (input_shape[2], input_shape[1]), label_name, False)
    # # 输出各个类别的精确率和召回率, 并输出为csv文件
    bodai_mistake_csv = config['bodai_mistake_csv']
    bodai_res_csv = config['bodai_res_csv']

    infer_from_dataload(model, val_list, (input_shape[2], input_shape[1]), label_name, bodai_mistake_csv,bodai_res_csv , False)






