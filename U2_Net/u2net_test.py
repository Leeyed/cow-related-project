import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB
import time
import cv2
from aiModel import Classfy_model

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # predict_np = np.where(predict_np < 0.1, 0, 1).astype(np.uint8)

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def save_output_mask(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    predict_np_ = np.where(predict_np < 0.5, 0, 1).astype(np.uint8)

    cv_img = cv2.imread(image_name)
    mask_np = cv2.resize(predict_np_, (cv_img.shape[1], cv_img.shape[0]))
    mask_np_1 = mask_np[:, :, np.newaxis]
    mask_np_3 = np.concatenate((mask_np_1, mask_np_1, mask_np_1), axis=-1)
    new_img = mask_np_3 * cv_img
    cv2.namedWindow("mask_img", cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow("mask_img", new_img)
    cv2.waitKey(0)

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    class_label = image_name.split(os.sep)[-2]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)
    #
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    class_label_path = os.path.join(d_dir, class_label) + os.sep
    if not os.path.exists(class_label_path):
        os.makedirs(class_label_path)

    # cv2.imwrite(class_label_path + imidx + '.jpg', new_img)
    #
    # imo.save(class_label_path + imidx + '.png')

import copy
def get_line_points_pic(image_name, pred, size, classfyModel):

    def cnt_area(contour):
        xmin = np.min(contour[:, :, 0])
        ymin = np.min(contour[:, :, 1])
        xmax = np.max(contour[:, :, 0])
        ymax = np.max(contour[:, :, 1])
        return np.sqrt(np.square(xmax - xmin) + np.square(ymax - ymin))

    def points3(contour):
        def takesecond(S):
            return S[1]

        def takefirst(S):
            return S[0]

        xmin = np.min(contour[:, :, 0])
        ymin = np.min(contour[:, :, 1])
        xmax = np.max(contour[:, :, 0])
        ymax = np.max(contour[:, :, 1])
        points = copy.deepcopy(contour[:, 0, :]).tolist()
        if ymax - ymin > xmax - xmin:
            points.sort(key=takesecond)
        else:
            points.sort(key=takefirst)
        length = len(points)
        return [points[int(length / 4)], points[int(length / 2)], points[int(3 * length / 4)]]

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    predict_np_ = np.where(predict_np < 0.5, 0, 1).astype(np.uint8)

    cv_img = cv2.imread(image_name)
    img_show = copy.deepcopy(cv_img)
    ori_w, ori_h = cv_img.shape[1], cv_img.shape[0]


    mask_np = cv2.resize(predict_np_, (cv_img.shape[1], cv_img.shape[0]))
    points = []
    cut_imgs = []
    try:
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cnt_area, reverse=True)
    for i in range(3):
        try:
            mask = np.zeros(cv_img.shape, dtype=np.uint8)
            cv2.drawContours(mask, contours, i, (1, 1, 1), -1)
            mask_pic = mask * cv_img
            for a in points3(contours[i]):
                # cX = int(np.median(contours[i][:, 0, 0]))
                # cY = int(np.median(contours[i][:, 0, 1]))
                # points.append([cX, cY])
                cX = a[0]
                cY = a[1]

                point_x0 = max(0, cX - int(size / 2))
                point_y0 = max(0, cY - int(size / 2))
                point_x1 = min(ori_w, cX + int(size / 2))
                point_y1 = min(ori_h, cY + int(size / 2))

                point_img_cut = mask_pic[int(point_y0): int(point_y1), int(point_x0): int(point_x1), :]
                obj, classfy_score = classfyModel.forward(point_img_cut)

                cv2.rectangle(img_show, (int(point_x0), int(point_y0)),
                              (int(point_x1), int(point_y1)),
                              [255, 255, 0], 5)
                cv2.putText(img_show, obj, (int(point_x0), int(point_y0) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

                # cv2.namedWindow("point_img_cut", cv2.WND_PROP_FULLSCREEN)
                # cv2.namedWindow("cv_img", cv2.WND_PROP_FULLSCREEN)
                # cv2.imshow("point_img_cut", point_img_cut)
                # cv2.imshow("cv_img", cv_img)
                # cv2.waitKey(0)
        except:
            break
    # cv2.namedWindow("img_show", cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow("point_img_cut", point_img_cut)
    # cv2.imshow("img_show", img_show)
    cv2.waitKey(0)


def get_dirs_all_data(imgdir):
    """
    获取imgdir下所有label内的图片, 返回所有图片的路径
    """
    img_list = []
    for each in os.listdir(imgdir):
        # 去除backgrond, wxqy
        if each == 'background' or each == 'wxqy' or each == 'hscfxpmgp' or each == 'hsbq':
            continue

        sub_dir = os.path.join(imgdir, each)
        files = os.listdir(sub_dir)
        paths = []
        for each_file in files:
            # 去除 'Thumbs.db' 文件
            if 'Thumbs.db' in each_file: continue
            paths.append(os.path.join(sub_dir, each_file))
        img_list.extend(paths)
    return img_list


# body align
import numpy
import math
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])

def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    return dst

def judge_angle(tmp_box):
    box = []
    x1 = tmp_box[2]
    y1 = tmp_box[3]
    x2 = tmp_box[4]
    y2 = tmp_box[5]
    x3 = tmp_box[6]
    y3 = tmp_box[7]
    ans = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if ans > 0:
        print('shun')
        return tmp_box
    elif ans < 0:
        # 逆时针的时候，2和4点互换
        box.append(tmp_box[0])
        box.append(tmp_box[1])
        box.append(tmp_box[6])
        box.append(tmp_box[7])
        box.append(tmp_box[4])
        box.append(tmp_box[5])
        box.append(tmp_box[2])
        box.append(tmp_box[3])
        print('ni')
        return box
    return box

import copy
def save_output_mask_rect(image_name, pred, d_dir):
    def cnt_area(contour):
        xmin = np.min(contour[:, :, 0])
        ymin = np.min(contour[:, :, 1])
        xmax = np.max(contour[:, :, 0])
        ymax = np.max(contour[:, :, 1])
        return np.sqrt(np.square(xmax - xmin) + np.square(ymax - ymin))

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np_ = np.where(predict_np < 0.5, 0, 1).astype(np.uint8)

    try:
        # @para: 第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法
        # RETR_LIST 从解释的角度来看，这中应是最简单的。它只是提取所有的轮廓，而不去创建任何父子关系。
        # RETR_EXTERNAL 如果你选择这种模式的话，只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉。
        # RETR_CCOMP 在这种模式下会返回所有的轮廓并将轮廓分为两级组织结构。
        # RETR_TREE 这种模式下会返回所有轮廓，并且创建一个完整的组织结构列表。它甚至会告诉你谁是爷爷，爸爸，儿子，孙子等。

        # cv2.CHAIN_APPROX_NONE，，表示边界所有点都会被储存
        # cv2.CHAIN_APPROX_SIMPLE 会压缩轮廓，将轮廓上冗余点去掉，比如说四边形就会只储存四个角点。
        contours, _ = cv2.findContours(predict_np_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv2.findContours(predict_np_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # temp = np.ones(predict_np_.shape, np.uint8) * 255
    # # 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
    # cv2.drawContours(temp, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('1', temp)

    contours.sort(key=cnt_area, reverse=True)
    # 去除微小的mask， 选取最大的
    new_predict_np_ = np.zeros(predict_np_.shape).astype(np.uint8)
    cv2.drawContours(new_predict_np_, contours, 0, (1, 1, 1), -1)

    # cv2.imshow('0: original image', new_predict_np_)

    # predict_np_125 = np.where(predict_np < 0.5, 125, 0).astype(np.uint8)

    cv_img = cv2.imread(image_name)
    # cv2.imshow('1: original image', cv_img)
    mask_np = cv2.resize(new_predict_np_, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 0,1 肉眼看不出来
    # cv2.imshow('2: mask image', mask_np)
    # mask_np_125 = cv2.resize(predict_np_125, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 求mask的最小外接矩形
    try:
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cnt_area, reverse=True)
    area = cv2.contourArea(contours[0])  # 计算面积
    rect = cv2.minAreaRect(contours[0])
    boxs = np.int0(cv2.boxPoints(rect))  # 计算最小外接矩形顶点
    # print(boxs)

    mask_np_125 = np.where(mask_np == 0, 125, 0).astype(np.uint8)
    # cv2.imshow('3: mask_np_125 image', mask_np_125)
    mask_np_1 = mask_np[:, :, np.newaxis]
    mask_np_1_125 = mask_np_125[:, :, np.newaxis]
    mask_np_3 = np.concatenate((mask_np_1, mask_np_1, mask_np_1), axis=-1)
    mask_np_3_125 = np.concatenate((mask_np_1_125, mask_np_1_125, mask_np_1_125), axis=-1)
    new_img = mask_np_3 * cv_img + mask_np_3_125
    # cv2.imshow('4: new_img image', new_img)

    ori_w, ori_h = new_img.shape[1], new_img.shape[0]
    # max_len = int(math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2))

    # new_img_ = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    horizon = False
    rotated_box = rect
    # center, size, angle
    # (x,y, (h,w),  x
    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    # print(center, size, angle)
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # print(center, size, angle)

    # 以mask旋转框的中心点，为新图片的中心点进行填充
    new_w, new_h = int(size[0]), int(size[1])
    c_x, c_y = int(center[0]), int(center[1])
    # if new_w > new_h:
    #     length = new_w
    # else:
    #     length = new_h
    length = int(math.sqrt((new_h) ** 2 + (new_w) ** 2))
    # w
    new_img_ = copy.deepcopy(new_img)
    center_x = c_x
    # left
    if c_x < length/2:
        tmp = int(length/2) - c_x
        new_img_ = cv2.copyMakeBorder(new_img_, 0, 0, tmp, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])
        center_x = int(length/2)
    # right
    if length/2 > (ori_w - c_x):
        tmp = int(length / 2 - (ori_w - c_x))
        new_img_ = cv2.copyMakeBorder(new_img_, 0, 0, 0, tmp, cv2.BORDER_CONSTANT, value=[125, 125, 125])

    # up
    center_y = c_y
    if c_y < length/2:
        tmp = int(length / 2) - c_y
        new_img_ = cv2.copyMakeBorder(new_img_, tmp, 0, 0, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])
        center_y = int(length / 2)

    # right
    if length / 2 > (ori_h - c_y):
        tmp = int(length / 2 - (ori_h - c_y))
        new_img_ = cv2.copyMakeBorder(new_img_, 0, tmp, 0, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])




    print(angle)
    #
    # if horizon:
    #
    #     if size[0] < size[1]:
    #         angle -= 270
    #         w = size[1]
    #         h = size[0]
    #     else:
    #         w = size[0]
    #         h = size[1]
    #     size = (w, h)

    height, width = new_img_.shape[0], new_img_.shape[1]
    if angle < -45:
        angle = angle + 90
        size = (size[1], size[0])
    center = (center_x, center_y)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # if size[0] > size[1]:
    new_w = int(math.sqrt((width) ** 2 + (height) ** 2))
    img_rot = cv2.warpAffine(new_img_, M, (width, height))

    img_crop = cv2.getRectSubPix(img_rot, size, center)
    # print(size)

    # tmp_box = [int(boxs[1][1]), int(boxs[1][1]), int(boxs[2][0]), int(boxs[2][1]),
    #            int(boxs[3][0]), int(boxs[3][1]), int(boxs[0][0]), int(boxs[0][1])]
    # # min_x = min(tmp_box[0], tmp_box[2], tmp_box[4], tmp_box[6])
    # #
    # #
    # # new_tmp_box = []
    # # for i in range(4):
    # #     if tmp_box[2*i] == min_x
    #
    # box = tmp_box
    #
    # cnt = [
    #     [[box[0], box[1]]],
    #     [[box[2], box[3]]],
    #     [[box[4], box[5]]],
    #     [[box[6], box[7]]]
    # ]
    # ori_point = [
    #     [box[0], box[1]],
    #     [box[2], box[3]],
    #     [box[4], box[5]],
    #     [box[6], box[7]]
    # ]
    #
    # width = int(math.sqrt((box[0] - box[2]) ** 2 + (box[1] - box[3]) ** 2))
    # height = int(math.sqrt((box[0] - box[6]) ** 2 + (box[1] - box[7]) ** 2))
    # coord_point = [
    #     [box[0], box[1]],
    #     [box[0] + width, box[1]],
    #     [box[0] + width, box[1] + height],
    #     [box[0], box[1] + height]
    # ]
    # warped = warp_im(new_img, ori_point, coord_point)
    # clip_img = warped[box[1]:(box[1] + height), box[0]:(box[0] + width), :]





    # cv2.namedWindow("img_rot", cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow("img_crop", img_crop)
    # cv2.imshow("img_rot", img_rot)
    #
    # # cv2.drawContours(new_img, [box], 0, (255, 0, 0), 1)
    # cv2.namedWindow("mask_img", cv2.WND_PROP_FULLSCREEN)
    # cv2.imshow("mask_img", new_img_)
    # cv2.waitKey(0)

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    class_label = image_name.split(os.sep)[-2]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)
    #
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    class_label_path = os.path.join(d_dir, class_label) + os.sep
    if not os.path.exists(class_label_path):
        os.makedirs(class_label_path)

    # cv2.imwrite(class_label_path + imidx + '.jpg', new_img)
    # 根据宽高看是否旋转90度
    img_crop_w, img_crop_h = img_crop.shape[1], img_crop.shape[0]
    if img_crop_w > img_crop_h:
        img_crop = np.rot90(img_crop)
    # cv2.imshow("img_crop", img_crop)
    # cv2.waitKey()
    cv2.imwrite(class_label_path + imidx + '.jpg', img_crop)

    #
    # imo.save(class_label_path + imidx + '.png')


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp
    image_dir = r'/home/liusheng/data/NFS120/cow/zzb_cow/cow_body_cam22_backup'
    prediction_dir = r'/home/liusheng/data/NFS120/cow/zzb_cow/cow_body_cam22_backup_sameway'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    # 新训练的牛体
    model_dir = './saved_models/u2net/u2net_bce_itr_291500_train_0.109425_tar_0.007453_body20210420_size320.pth'

    img_name_list = get_dirs_all_data(image_dir)

    print("imgs num: ", len(img_name_list))

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),  # 480 == 线缝   320 == 牛体
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # classfy model
    # model_path = '/home/zhouzhubin/workspace/project/pytorch/cow_bodai_classfy/checkpoints/xianfeng_20210325/60/model.pth'
    # class_path = '/home/zhouzhubin/workspace/project/pytorch/cow_bodai_classfy/xianfeng_class_num_20210325.txt'
    # input_size = [3, 128, 128]
    # classfyModel = Classfy_model(model_path, class_path, input_size)

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        t1 = time.time()
        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        # print("infer time: ", time.time() - t1)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        save_output_mask_rect(img_name_list[i_test], pred, prediction_dir)
        # exit()
        # get_line_points_pic(img_name_list[i_test], pred, 128, classfyModel)
        # time.sleep(10)
        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()


