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

data_save_path = "./data/my_data/bcs_20200709_all_data.txt"
# bcs_20200603_test 是用上一批数据,来验证算法,新的数据处理方式,是否是真的影响模型的训练,还是因为新加入的数据的原因

data_save = open(data_save_path, "w")
id = 0

# batch_list = []
# for batch in open("./data/my_data/car_seat_batch_list_2020_04_18.txt").readlines():
#
#     batch_list.append(batch.split("\n")[0])
# 头戴式的拍摄
# batch_list = ["gmnnc-bs-2020042601", "gmnnc-bs-2020042801", "gmnnc-bs-2020042802", "gmnnc-bs-2020042803",
#               "gmnnc-bs-2020043001", "gmnnc-bs-2020043002", "gmnnc-bs-2020043003", "gmnnc-bs-2020043004",
#               "gmnnc-bs-2020043005", "gmnnc-bs-2020050701", "gmnnc-bs-2020050702", "gmnnc-bs-2020050703",
#               "gmnnc-bs-2020050704", "gmnnc-bs-2020050705", "gmnnc-bs-2020050901", "gmnnc-bs-2020050902",
#               "gmnnc-bs-2020050903", "gmnnc-bs-2020050904", "gmnnc-bs-2020050905", "gmnnc-bs-2020050906",
#               "gmnnc-bs-2020051204",
#               ]
#　固定式的拍摄
# batch_list = ["gmnnc-bs-2020051801", "gmnnc-bs-2020051802", "gmnnc-bs-2020051803",
#               "gmnnc-bs-2020051804"]
# batch_list = ["gmnnc-bs-2020052101", "gmnnc-bs-2020052102", "gmnnc-bs-2020052103"]
# 0628
# batch_list = ["062801", "062802", "062803", "070101"]

# 20200705 训练
# batch_list = ["gmnnc-bs-2020042601", "gmnnc-bs-2020042801", "gmnnc-bs-2020042802", "gmnnc-bs-2020042803",
#               "gmnnc-bs-2020043001", "gmnnc-bs-2020043002", "gmnnc-bs-2020043003", "gmnnc-bs-2020043004",
#               "gmnnc-bs-2020043005", "gmnnc-bs-2020050701", "gmnnc-bs-2020050702", "gmnnc-bs-2020050703",
#               "gmnnc-bs-2020050704", "gmnnc-bs-2020050705", "gmnnc-bs-2020050901", "gmnnc-bs-2020050902",
#               "gmnnc-bs-2020050903", "gmnnc-bs-2020050904", "gmnnc-bs-2020050905", "gmnnc-bs-2020050906",
#               "gmnnc-bs-2020051204", "gmnnc-bs-2020051801", "gmnnc-bs-2020051802", "gmnnc-bs-2020051803",
#               "gmnnc-bs-2020051804", "gmnnc-bs-2020052101", "062801", "062802", "062803", "070101"
#               ]

# 20200709 目前所有的标注数据
batch_list = ["gmnnc-bs-2020042601", "gmnnc-bs-2020042801", "gmnnc-bs-2020042802", "gmnnc-bs-2020042803",
              "gmnnc-bs-2020043001", "gmnnc-bs-2020043002", "gmnnc-bs-2020043003", "gmnnc-bs-2020043004",
              "gmnnc-bs-2020043005", "gmnnc-bs-2020050701", "gmnnc-bs-2020050702", "gmnnc-bs-2020050703",
              "gmnnc-bs-2020050704", "gmnnc-bs-2020050705", "gmnnc-bs-2020050901", "gmnnc-bs-2020050902",
              "gmnnc-bs-2020050903", "gmnnc-bs-2020050904", "gmnnc-bs-2020050905", "gmnnc-bs-2020050906",
              "gmnnc-bs-2020051204", "gmnnc-bs-2020051801", "gmnnc-bs-2020051802", "gmnnc-bs-2020051803",
              "gmnnc-bs-2020051804", "gmnnc-bs-2020052101", "gmnnc-bs-2020052102", "gmnnc-bs-2020052103",
              "062801", "062802", "062803", "070101"
              ]

# 固定摄像头下的数据
# batch_list = ["gmnnc-bs-2020051801", "gmnnc-bs-2020051802", "gmnnc-bs-2020051803",
#               "gmnnc-bs-2020051804", "gmnnc-bs-2020052101", "gmnnc-bs-2020052102", "gmnnc-bs-2020052103",
#               "062801", "062802", "062803", "070101"]


# 汇总0427-0531固定式视频下的数据
# batch_list = ["gmnnc-bs-2020051801", "gmnnc-bs-2020051802", "gmnnc-bs-2020051803",
#               "gmnnc-bs-2020051804", "gmnnc-bs-2020052101", "gmnnc-bs-2020052102", "gmnnc-bs-2020052103", "062801", "062802", "062803", "070101"
#               ]
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
