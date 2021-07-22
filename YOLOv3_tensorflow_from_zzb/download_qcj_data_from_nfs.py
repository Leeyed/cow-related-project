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


project_name = "gmnnc_qcj"

data_save_path = "./data/my_data/qcj_20200827_data.txt"
# bcs_20200603_test 是用上一批数据,来验证算法,新的数据处理方式,是否是真的影响模型的训练,还是因为新加入的数据的原因

data_save = open(data_save_path, "w")
id = 0

# 20200709 目前所有的标注数据
batch_list = ["qcj_20200821", "qcj_20200821_2", "qcj_20200821_1"]
for batch in batch_list:
    data = requestWithSign("aiadminapi/GetImageRecord/listByWhere", {"batchNo":batch,"pageSize":10000,"belongBusiness":project_name, "status": 1})#"handler_status":2
    # str = json.loads(html)
    # list_key = []
    print(data)
    print(batch)
    for children in data:
        # img_path = "/mnt/AIdata/images4code2/" + children["path"]
        img_path = children["path"]
        boxes_list = children["imageLabelRecordList"]
        boxes = []
        labels = []

        for box in boxes_list:
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
            # if labels[i] != "waz072714hsslqj":
            line = line + " %s %d %d %d %d" % (labels[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
        data_save.write(line)
        data_save.write("\n")
        id = id + 1

data_save.close()


