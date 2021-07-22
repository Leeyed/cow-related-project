import urllib.request
import os
import json
import cv2

import requests
import time
import hashlib


def requestWithSign(path, params, host="http://openapi.gmmsj.com/"):
    fixed_params = {
        "merchant_name": "AI_ADMIN",
        "timestamp": str(int(time.time())),
        "signature_method": "MD5"
    }

    params.update(fixed_params)

    url = host + path
    params["signature"] = sign(params)

    print("{}?{}".format(url, params))
    response = requests.get(url=url, params=params)
    response.raise_for_status()
    result = response.json()
    #     if result['return_code'] != 0:
    #         raise RequestInteralError(**result)
    print(result)
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
    print(paraSign)
    sigValue = hashlib.md5(paraSign).hexdigest()

    return sigValue


def getHttp(url):
    page = urllib.request.urlopen(url)
    str = page.read()
    return str


project_name = "gmnnc-bs"
# 奶台号码牌数据
# data_save_path = "./data/my_data/bcs_202007011_num_A.txt"  # 只保存A种型号的号码牌
# data_save_path = "data/my_data/bcs_202007026_alldata_haikang.txt"  # 所有海康数据的
data_save_path = "data/my_data/bcs_20200805_num.txt"  #
model = 'A'
#
'''
bcs_20200709_all_data.txt  gopro摄像头所拍摄的所有数据

bcs_202007016_alldata_cow_num.txt 是新的海康摄像头下，采集的数据，其中包含号码牌和牛体
'''
#
# data_save_path = "./data/my_data/bcs_20200708_nt_data_B.txt"
# model = 'B'

data_save = open(data_save_path, "w")
id = 0

# batch_list = []
# for batch in open("./data/my_data/car_seat_batch_list_2020_04_18.txt").readlines():


# 固定摄像头下的数据
# batch_list = ["bcs_nbr_070701"]
# batch_list = ["bcs_nbr_070701", "bcs070801"]
# batch_list = ["bcs070902"]
# batch_list = ["bcs_nbr_070701", "bcs070801", "bcs070902"]
# batch_list = ["bcstest0709"]
# batch_list = ["bcstest0709", "bcs070903"]
# batch_list = ["bcs_nbr_070701", "bcs070801", "bcs070902", "bcs070903", "bcs070901", "bcs070904"]
# batch_list = ["bcs_nbr_070701", "bcs070801", "bcs070902", "bcs070903", "bcs070901", "bcs070904", "bcs072101", "bcs072102"]
# batch_list = ["bcs_nbr_070701", "bcsnum080501"]
batch_list = ["bcsnum080501"]


for batch in batch_list:
    data = requestWithSign("aiadminapi/GetImageRecord/listByWhere", {"batchNo":batch,"pageSize":10000,"belongBusiness":project_name, "status": 1, "handlerStatus": 2})#"handler_status":2
    # str = json.loads(html)
    # list_key = []
    print(data)
    for children in data:
        # img_path = "/mnt/AIdata/images4code2/" + children["path"]
        img_path = children["path"]
        boxes_list = children["imageLabelRecordList"]
        boxes = []
        labels = []

        for box in boxes_list:
            label = box["labelType"]
            if (label[0] == 'n' and label[4] == model) or label[0] == 'b':  # A or B or label = 'bcs'
                x0, y0, x1, y1 = box["leftTopX"], box["leftTopY"], box["rightBottomX"], box["rightBottomY"]
                # if box["labelType"] != "waz072714hsslqj":
                labels.append(box["labelType"])
                boxes.append([x0, y0, x1, y1])
        right = True
        # for label in labels:
        #     if label not in jswq:
        #         right = False
        # if not right:
        #     print("find not right one")
        # if len(boxes) == 4 and right:
        line = img_path
        for i in range(len(boxes)):
            # 将数据类型为 牛体评分 号码牌 ....
            # if labels[i] != "waz072714hsslqj":
            if labels[i].startswith('bs'):
                line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

        for i in range(len(boxes)):
            # 将数据类型为 牛体评分 号码牌 ....
            # if labels[i] != "waz072714hsslqj":
            if labels[i].startswith('num'):
                line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
        data_save.write(line)
        data_save.write("\n")
        id = id + 1

data_save.close()
