res_w = 768

res_h = 1024

if res_w <= 1024 and res_h <= 768:
    print("s")
elif res_w <= 2048 and res_h <= 1600:
    print("m")
elif res_w <= 3008 and res_h <= 2016:
    print("b")

import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

boxes_select = np.asarray(a[0], np.int32)

x0, y0, x1, y1 = boxes_select

print("**********")

import os
import shutil


def copy_pic_from_txt(txt, save_dir, ori_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_dir = '/home/zhouzhubin/sjht_data/'
    data = open(txt, 'r')
    line_datas = data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split('%')

        img_name = line_list[0].split('/')[-1][:-4]
        print(img_name)
        ori_img_name = "%s.jpg" % img_name
        shutil.copy(os.path.join(ori_dir, "ori_pic", ori_img_name), os.path.join(save_dir, "ori_pic", ori_img_name))

        mask_img_name = "%s.png" % img_name
        shutil.copy(os.path.join(ori_dir, "0_255_pic", mask_img_name),
                    os.path.join(save_dir, "0_255_pic", mask_img_name))

    data.close()


txt_name = "20210226_cow_data_train.txt"
save_dir = "/home/zhouzhubin/workspace/project/datasets/u2net/20210225_data/train"
ori_dir = "/home/zhouzhubin/workspace/project/datasets/u2net/20210225_data"

# copy_pic_from_txt(txt_name, save_dir, ori_dir)

txt_name = "20210226_cow_data_val.txt"
save_dir = "/home/zhouzhubin/workspace/project/datasets/u2net/20210225_data/val"
# copy_pic_from_txt(txt_name, save_dir, ori_dir)

import random


def split_train_val(src_dir, scal=0.2):
    val_dir = os.path.join(src_dir, 'val')
    val_ori_pic = os.path.join(val_dir, 'ori_pic')
    val_0_255_pic = os.path.join(val_dir, '0_255_pic')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

        os.makedirs(val_ori_pic)
        os.makedirs(val_0_255_pic)

    train_dir = os.path.join(src_dir, 'train')
    train_ori_pic = os.path.join(train_dir, 'ori_pic')
    train_0_255_pic = os.path.join(train_dir, '0_255_pic')
    ori_pic_list = os.listdir(train_ori_pic)
    split_num = int(scal * len(ori_pic_list))

    random.shuffle(ori_pic_list)
    for i, each in enumerate(ori_pic_list):

        each_name = each[:-4]
        print(each_name)
        if i < split_num:
            shutil.move(os.path.join(train_ori_pic, each), os.path.join(val_ori_pic, each))
            shutil.move(os.path.join(train_0_255_pic, each_name + '.png'),
                        os.path.join(val_0_255_pic, each_name + '.png'))


src_dir = "/home/zhouzhubin/workspace/project/datasets/u2net/20210315_data"

# split_train_val(src_dir)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


print(a[:, 0].max())
print(a[:, 0].min())

def find_bbox_label(bb_label_data):
    bboxs = []
    labels = []
    data_len = len(bb_label_data)
    for i in range(data_len//5):  # label bbox
        bboxs.append((float(bb_label_data[i * 5 + 1]),
                      float(bb_label_data[i * 5 + 2]),
                      float(bb_label_data[i * 5 + 3]),
                      float(bb_label_data[i * 5 + 4])))
        labels.append(bb_label_data[i * 5])
    return bboxs, labels


from PIL import Image, ImageDraw, ImageFont

import cv2
def check_cow_body_data(txt):
    txt_data = open(txt, 'r')

    line_datas = txt_data.readlines()
    num = 0
    size_list = []
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        # line_list = line_list[1:]  # 带id
        img_name1 = line_list[0]
        img_name2 = line_list[1]

        img1 = cv2.imread(img_name1)
        img2 = cv2.imread(img_name2)
        print("img1 size: %s, img2 size: %s" % (str(img1.shape), str(img2.shape)))
        size_list.append(img1.shape)
        if img1.shape != img2.shape:
            num += 1
            print("####")

        bb_label_data = line_list[2:]
        bboxs, labels = find_bbox_label(bb_label_data)
        cv2img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)
        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        # font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        font = ImageFont.truetype("/usr/share/fonts/wps-office/simhei.ttf", 30, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
        for i, bbox in enumerate(bboxs):

            pointx = max(0, int(bbox[0]) - 30)
            pointy = max(0, int(bbox[1]) - 30)
            draw.text((pointx, pointy), labels[i], (255, 0, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

        # PIL图片转cv2 图片
        cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        for i, bbox in enumerate(bboxs):
            cv2.rectangle(cv2charimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [255, 255, 0], 5)


        cv2.namedWindow("img1", cv2.WND_PROP_FULLSCREEN)
        cv2.namedWindow("img2", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("img1", cv2charimg)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)
    print("size is diffient num: ", num)
    print("size: ", list(set(size_list)))

txt = 'cow_body_model_20210318_270.txt'

# check_cow_body_data(txt)

txt = '20210312_fabric_data.txt'
txt = 'new_txt.txt'

check_cow_body_data(txt)
