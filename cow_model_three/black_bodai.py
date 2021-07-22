import time
import cv2
import os
import random
import shutil


from aiModel import Ai_model, Ai_model_three

# 确保取到真正的脚本目录，不要用sys.path[0]
TASK_PY_PATH = os.path.split(os.path.realpath(__file__))[0]

config_path = os.path.join(TASK_PY_PATH, 'cfg_bodai.yaml')

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

def balck_neckband(aimodel, ori_dir):
    all_dir_list = []
    all_img_list = []
    for folder in os.listdir(ori_dir):
        [all_dir_list.append(os.path.join(ori_dir, folder))]
    for folder in all_dir_list:
        imglist = [os.path.join(folder, pic) for pic in os.listdir(folder)]
        all_img_list.extend(imglist)

    for i, img_path in enumerate(all_img_list):
        if not img_path.endswith('.jpg'):
            continue
        detect_classfy_move_pic(aimodel, img_path)
    return


if __name__ == '__main__':
    aimodel = Ai_model_three(config_path)
    ori_dir = '/home/liusheng/data/NFS120/cow/zzb_cow/cow_body_cam22_backup'
    # save_dir = '/home/liusheng/data/NFS120/cow_video/cowid/suqian-14-1119/DVR_Examiner_Export_2020-11-25 174020_Job_0002/2020-11-19/select_upload_data/test_liusheng/'
    balck_neckband(aimodel, ori_dir)