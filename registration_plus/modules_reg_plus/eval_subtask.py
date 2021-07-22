import os
import time
import pickle
import torch
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torchvision import transforms as T

from modules_reg_plus.utils_reg import get_model_keywords, prepare_image


@torch.no_grad()
def save_infer_data(data_dir, save_name, model, config):
    input_shape = config['input_shape']
    img_list = []
    for label in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, label)
        imgs = os.listdir(sub_dir)
        for img in imgs:
            # 去除 'Thumbs.db' 文件
            if not img.endswith('.jpg'): continue
            if '副本' in img: continue
            img_list.append(os.path.join(sub_dir, img))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean, std)])
    batch_size = 256
    save_feature_data = {}
    cnt = 0
    batch_data = []
    batch_keys = []

    while (cnt < len(img_list)):
        image_path = img_list[cnt]
        input_data = prepare_image(image_path, (input_shape[2], input_shape[1]), transform)
        input_data = input_data.unsqueeze(0).to(config['device'])
        batch_keys.append(image_path)
        batch_data.append(input_data)

        cnt += 1
        # python [int] infinity
        if cnt%batch_size == 0 or cnt == len(img_list):
            t1 = time.time()
            batch_data = torch.cat(batch_data)
            features = model(batch_data)
            print('batch infer time: %f' % (time.time() - t1))

            for j, feature_key in enumerate(batch_keys):
                save_feature_data[feature_key] = features[j].cpu().numpy()

            batch_data = []
            batch_keys = []

    # 将输出的保存为pkl
    # print('exit in eval_subtask, save_infer_data')
    # exit()
    save_data_output = open('feature.pkl', 'wb')
    pickle.dump(save_feature_data, save_data_output)
    save_data_output.close()


def get_ref_test_feature(config, model):
    ref_dir = config['ref_dir']
    ref_data_pkl = os.path.join(config['save_path'],
                                config['checkpoint_subtitle'],
                                "{date}_{model}_{ref}.pkl".format(
                                    date=time.strftime("%Y%m%d"),
                                    model=get_model_keywords(config['train_root'].split('/')[-1]),
                                    ref=ref_dir.split('/')[-1]))
    if not os.path.exists(ref_data_pkl):
        save_infer_data(ref_dir, ref_data_pkl, model, config)

    # test data pickle
    test_dir = config['test_dir']
    test_data_pkl = os.path.join(config['save_path'],
                                 config['checkpoint_subtitle'],
                                 "{date}_{model}_test.pkl".format(
                                     date=time.strftime("%Y%m%d"),
                                     model=get_model_keywords(config['train_root'].split('/')[-1])))
    if not os.path.exists(test_data_pkl):
        save_infer_data(test_dir, test_data_pkl, model, config)

    f = open(ref_data_pkl, 'rb')
    ref_feature_dict = pickle.load(f)
    f.close()
    f = open(test_data_pkl, 'rb')
    test_feature_dict = pickle.load(f)
    f.close()
    # cpu -> gpu
    for key, value in ref_feature_dict.items():
        ref_feature_dict[key] = torch.from_numpy(value).to(config['device'])
    for key, value in test_feature_dict.items():
        test_feature_dict[key] = torch.from_numpy(value).to(config['device'])

    return ref_feature_dict, test_feature_dict


def gen_sample_pair(config):
    url = config['test_dir']
    file_path = os.path.join(config['save_path'], config['checkpoint_subtitle'], str(config['epochs']), 'img_pair.txt')
    random.seed(777)

    # 按照label分类 图片
    imgs_by_class = []
    for dirpath, dirnames, filenames in os.walk(url):
        for dir in dirnames:
            imgs = os.listdir(os.path.join(dirpath, dir))
            dir_imgs = [os.path.join(dirpath, dir, img) for img in imgs]
            dir_imgs = list(filter(lambda x: x.endswith('.jpg'), dir_imgs))
            imgs_by_class.append(dir_imgs)
    print('len(imgs_by_class)', len(imgs_by_class))

    # TP samples
    with open(file_path, 'w') as f:
        for cls_imgs in imgs_by_class:
            cmb = list(itertools.combinations(cls_imgs, 2))
            cmb = random.sample(cmb, min(len(cmb), 1000))
            for cmb_item in cmb:
                f.write("%s,%s,%s \n" % (cmb_item[0], cmb_item[1], 1))

    # F samples
    with open(file_path, 'a') as f:
        # for cls_imgs in imgs_by_class:
        two_cls = list(itertools.combinations(imgs_by_class, 2))
        print('len(two_cls)', len(two_cls))
        for cls1, cls2 in two_cls:
            cmb = list(itertools.product(cls1, cls2))
            cmb = random.sample(cmb, min(len(cmb), 1000))
            # print(cls1, cls2, 'complete')
            for cmb_item in cmb:
                f.write("%s,%s,%s \n" % (cmb_item[0], cmb_item[1], 0))


@torch.no_grad()
def cal_sim_pairs(config, model):
    _,test_feature_dict = get_ref_test_feature(config, model)

    file_path = os.path.join(config['save_path'], config['checkpoint_subtitle'], str(config['epochs']), 'img_pair.txt')
    with open(file_path, 'r') as f:
        data = f.readlines()
        print('total sample pairs:', len(data))
    pairs = [d.strip().split(',') for d in data]

    t1 = time.time()
    predicts, labels = [],[]
    f1_mat,f2_mat=[],[]
    print('start calculate sim')
    batch_size=256
    for i,line in enumerate(pairs):
        try:
            f1 = test_feature_dict[line[0]]
            f2 = test_feature_dict[line[1]]
        except:
            print(line[0], 'error')
            print(line[1], 'error')
            continue
        f1_mat.append(f1)
        f2_mat.append(f2)
        labels.append(int(line[2]))
        if (i+1)%batch_size == 0 or (i+1) == len(pairs):
            f1_mat = torch.stack(f1_mat)
            f2_mat = torch.stack(f2_mat)
            predicts.extend(list(torch.cosine_similarity(f1_mat, f2_mat).cpu().numpy().reshape(-1)))
            f1_mat = []
            f2_mat = []
            if random.random()<0.01:
                print((i+1), 'finished')
    print('sim calculate time: %f' % (time.time() - t1))
    return labels, predicts



def get_th_from_roc(config, model):
    labels, predicts = cal_sim_pairs(config, model)
    fpr, tpr, threshold = roc_curve(labels, predicts)
    plt.figure(figsize=(15, 10))
    plt.plot(fpr, tpr)
    plt.xticks(rotation=90)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = os.path.join(config['save_path'], config['checkpoint_subtitle'])
    plt.savefig(os.path.join(path, 'test_sample_1000.jpg'))

    ret = []
    target_fprs = [0.1, 0.01, 0.001]
    for each in target_fprs:
        idx = np.argmin(abs(fpr - each))
        print(f'fpr: {fpr[idx]}, tpr: {tpr[idx]}, th: {threshold[idx]}')
        ret.append(threshold[idx])
    return ret
