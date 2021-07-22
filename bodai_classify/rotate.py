import time
import cv2
import os
import random
import shutil
import yaml
from PIL import Image

from aiModel import Ai_model, Ai_model_three, Classfy_model

# 确保取到真正的脚本目录，不要用sys.path[0]
TASK_PY_PATH = os.path.split(os.path.realpath(__file__))[0]
config_path = os.path.join(TASK_PY_PATH, 'cfg.yaml')

def detect_classfy_move_pic(model, imagePath):
    data, img_show = model.forward(imagePath)
    # print(data)
    # cv2.namedWindow("img_show", cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow('img_show', img_show)
    # cv2.waitKey(0)
    # cv2.destroyWindow("img_show")
    ### 这里可以修改保存结果图片
    # name = random.randint(1,100000)
    # name = str(name) + '.jpg'
    # cv2.imwrite(os.path.join('/home/liusheng/data/NFS120/zzb_cow/cow_body/imwrite', name), img_show)

    return data

def rotate(aimodel, ori_dir, dst_url):
    all_dir_list = []
    all_img_list = []
    uncertain_info = []
    for folder in os.listdir(ori_dir):
        [all_dir_list.append(os.path.join(ori_dir, folder))]
    for folder in all_dir_list:
        imglist = [os.path.join(folder, pic) for pic in os.listdir(folder)]
        all_img_list.extend(imglist)
    print('len(all_img_list)', len(all_img_list))
    for i, img_path in enumerate(all_img_list):
        if not img_path.endswith('.jpg'):continue
        # out: label, score
        print(i)
        img = cv2.imread(img_path)
        label, score = aimodel.forward(img)

        im = Image.open(img_path)
        if label=='ntcx':
            im = im.transpose(Image.ROTATE_180)
        if not os.path.exists(os.path.join(dst_url, img_path.split(os.sep)[-2])):
            os.makedirs(os.path.join(dst_url, img_path.split(os.sep)[-2]))
        im.save(os.path.join(dst_url, img_path.split(os.sep)[-2], img_path.split(os.sep)[-1]))
    return


if __name__ == '__main__':
    with open(config_path, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    print(config)
    # __init__(self, model_path, class_path, input_size, threshold=0.5):
    aimodel = Classfy_model(config['model_path'],
                            config['class_label'],
                            config['input_shape'],
                            config['threshold']) # th 在Classfy_model无应用，在aimodel3 or aimodel 中有作用
    rotate_url = config['rotate_url']
    dst_url = config['dst_url']
    rotate(aimodel, rotate_url, dst_url)