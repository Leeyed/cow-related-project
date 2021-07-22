"""读取服务端annotation"""
from random import shuffle
import urllib.request
import os
import json
import cv2
import sys
import time

import requests
import time
import hashlib
import shutil
import xml.etree.cElementTree as ET
from xml.dom.minidom import Document
import numpy as np
import math


def judge_angle(tmp_box):
    box = []
    x1 = tmp_box[2]
    y1 = tmp_box[3]
    x2 = tmp_box[4]
    y2 = tmp_box[5]
    x3 = tmp_box[6]
    y3 = tmp_box[7]
    ans = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)
    if ans > 0:
        # print('shun')
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
        # print('ni')
        return box
    return box


# def requestWithSign(path, params, host="http://openapi.gmmsj.com/"):
def requestWithSign(path, params, host="https://ai-openapi.gmm01.com/"):
    fixed_params = {
        "merchant_name": "AI_ADMIN",
        "timestamp": str(int(time.time())),
        "signature_method": "MD5"
    }

    params.update(fixed_params)

    url = host + path
    params["signature"] = sign(params)

    # print("{}?{}".format(url, params))
    response = requests.get(url=url, params=params)
    response.raise_for_status()
    result = response.json()
    #     if result['return_code'] != 0:
    #         raise RequestInteralError(**result)
    # print(result)
    if 'data' in result:
        result = result['data']
    else:
        result = None
    return result

def sign(params):
    sigKey = "8HM2NiElGzSIq9nNPtTW0ZH8Vk7YLWRB"
    sigValue = ""
    paraSign = ""
    sortData = {}

    sortData = sorted(params.items(), key=lambda asd: asd[0], reverse=False)

    for item in sortData:
        paraSign = paraSign + item[0] + "=" + str(item[1])

    paraSign = paraSign + sigKey
    paraSign = paraSign.encode()
    # print(paraSign)
    sigValue = hashlib.md5(paraSign).hexdigest()

    return sigValue


#print(urllib.__file__)
def getHttp(url):
    page = urllib.request.urlopen(url)
    str = page.read()
    return str

def get_segmetation_data_txt(batch, txt, name, handlerStatus=1, status=1, NFS_dir='/home/zhouzhubin/sjht_data/'):
    assert isinstance(batch, (list, tuple))
    assert isinstance(txt, str)

    data_save = open(txt, 'w')

    for once in batch:
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere",
                                    {"batchNo": once, "pageSize": 10000, "belongBusiness": name,
                                     "handlerStatus": handlerStatus, "status": status})

        for children in data:
            img_path = os.path.join(NFS_dir, children["path"])
            img_name = img_path.split('/')[-1]
            boxes_list = children["imageLabelPolygonExt"]
            if boxes_list == []:
                continue
            else:
                box_num = 0
                boxes = []
                for box in boxes_list:

                    labelType = box['labelType']
                    points_list = list(map(lambda x: [int(x["x"]), int(x["y"])],
                                           eval(box["polygonJson"]) if isinstance(box["polygonJson"], str) else box[
                                               "polygonJson"]))
                    x_list = [point[0] for point in points_list]
                    y_list = [point[1] for point in points_list]

                    xmin, ymin = max(min(x_list), 0), max(min(y_list), 0)
                    xmax, ymax = max(max(x_list), 0), max(max(y_list), 0)
                    if xmax - xmin <= 5 or ymax - ymin <= 5:
                        continue
                    boxes.append([xmin, ymin, xmax, ymax])
                # write txt
                line = children["path"]
                for i in range(len(boxes)):
                    # if labels[i] != "waz072714hsslqj":
                    line = line + " %s %d %d %d %d" % (
                    "bodai", boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                data_save.write(line)
                data_save.write("\n")

    data_save.close()


def get_segmetation_txt(batch, txt, name, handlerStatus=2, status=1, NFS_dir='/home/zhouzhubin/sjht_data/'):
    assert isinstance(batch, (list, tuple))
    assert isinstance(txt, str)

    data_save = open(txt, 'w')

    for once in batch:
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere",
                                    {"batchNo": once, "pageSize": 10000, "belongBusiness": name,
                                     "handlerStatus": handlerStatus, "status": status})

        for children in data:
            img_path = os.path.join(NFS_dir, children["path"])
            img_name = img_path.split('/')[-1]
            boxes_list = children["imageLabelPolygonExt"]
            if boxes_list == []:
                continue
            else:
                box_num = 0
                boxes = []
                line = children["path"]
                for box in boxes_list:

                    labelType = box['labelType']
                    points_list = list(map(lambda x: [int(x["x"]), int(x["y"])],
                                           eval(box["polygonJson"]) if isinstance(box["polygonJson"], str) else box[
                                               "polygonJson"]))
                    x_list = [point[0] for point in points_list]
                    y_list = [point[1] for point in points_list]
                    line = line + "#%s" % labelType
                    for i, x in enumerate(x_list):
                        line = line + " %d %d" % (x, y_list[i])

                data_save.write(line)
                data_save.write("\n")

    data_save.close()



def get_segmetation_data_pic(batch, save_dir, name, handlerStatus=2, status=1, NFS_dir='/home/zhouzhubin/sjht_data/'):
    assert isinstance(batch, (list, tuple))
    assert isinstance(save_dir, str)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for once in batch:
        data = requestWithSign("aiadminapi/GetImageRecord/listByWhere",
                                    {"batchNo": once, "pageSize": 10000, "belongBusiness": name,
                                     "handlerStatus": handlerStatus, "status": status})

        for children in data:
            img_path = os.path.join(NFS_dir, children["path"])
            img_name = img_path.split('/')[-1][:-4]
            boxes_list = children["imageLabelPolygonExt"]

            img = cv2.imread(img_path)
            ori_w, ori_h = img.shape[1], img.shape[0]

            if boxes_list == []:
                continue
            else:
                box_num = 0
                boxes = []
                for box in boxes_list:

                    labelType = box['labelType']
                    points_list = list(map(lambda x: [int(x["x"]), int(x["y"])],
                                           eval(box["polygonJson"]) if isinstance(box["polygonJson"], str) else box[
                                               "polygonJson"]))
                    mask = np.zeros([ori_h, ori_w, 3], np.uint8)
                    p = np.array(points_list)
                    cv2.fillPoly(mask, [p], (1, 1, 1))
                    # ori
                    ori_path = os.path.join(save_dir, "ori_pic")
                    if not os.path.exists(ori_path):
                        os.makedirs(ori_path)
                    ori_name = '%s.jpg' % img_name
                    cv2.imwrite(os.path.join(ori_path, ori_name), img)
                    # mask
                    mask_path = os.path.join(save_dir, "mask_pic")
                    if not os.path.exists(mask_path):
                        os.makedirs(mask_path)
                    mask_name = '%s.png' % img_name
                    # cv2.imwrite(os.path.join(mask_path, mask_name), mask)
                    print(mask_name)
                    # 0_255_pic
                    mask = np.zeros([ori_h, ori_w, 3], np.uint8)
                    p = np.array(points_list)
                    cv2.fillPoly(mask, [p], (255, 255, 255))
                    mask_path = os.path.join(save_dir, "0_255_pic")
                    if not os.path.exists(mask_path):
                        os.makedirs(mask_path)
                    mask_name = '%s.png' % img_name
                    cv2.imwrite(os.path.join(mask_path, mask_name), mask)


if __name__ == '__main__':
    batchlist = ["20210220_cow_body_inssegment"]
    save_dir = '/home/zhouzhubin/workspace/project/datasets/u2net/20210225_data'
    batchlist = ["20210220_cow_body_inssegment", "20210226_cow_body_inssegment"]
    save_dir = '/home/zhouzhubin/workspace/project/datasets/u2net/20210315_data'
    # get_segmetation_data_pic(batchlist, save_dir, "gmnnc_qcj")


    # 线缝的数据
    batchlist = ["20210318161914-568"]
    save_dir = '/home/zhouzhubin/sjht_data/Datas/zzb/xianfeng/20210322_data'
    # get_segmetation_data_pic(batchlist, save_dir, "sjht")

    txt = 'xianfeng20210323.txt'
    get_segmetation_txt(batchlist, txt, "sjht")
