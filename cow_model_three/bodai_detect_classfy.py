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
    # data, img_show = model.forward(imagePath)
    data = model.forward(imagePath)
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

### 裁剪图片
def cropPic(aimodel, ori_dir, save_dir, num):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_img_list = []
    all_dir_list = []
    for folder in os.listdir(ori_dir):
        [all_dir_list.append(os.path.join(ori_dir, folder))]
    print('all_dir_list', len(all_dir_list))

    for folder in all_dir_list:
        imglist = [os.path.join(folder, pic) for pic in os.listdir(folder)]
        all_img_list.extend(imglist)
    print('all_img_list', len(all_img_list))

    move_img_list = []  # 需要移动的图片
    ### bodai classification
    # for i, img_path in enumerate(all_img_list[:100]):
    #     if not img_path.endswith('.jpg'):
    #         continue
    #     data = detect_classfy_move_pic(aimodel, img_path)
    #     for i, single_box in enumerate(data):
    #         if single_box['label'] == "" or single_box['label'] == "blurry":
    #             continue
    #         move_img_list.append([single_box['label'], single_box['box'], img_path])
    #         if len(move_img_list) % 100 == 1:
    #                 print(len(move_img_list), 'of', i)
    #
    # ### copy bodai
    # for i, move_img in enumerate(move_img_list):
    #     img_class = move_img_list[i][0]
    #     x0, y0, x1, y1 = move_img_list[i][1]
    #     img_path = move_img_list[i][2]
    #     if not os.path.exists(os.path.join(save_dir,img_class)):
    #         os.makedirs(os.path.join(save_dir,img_class))
    #     cow_body_img = cv2.imread(img_path)
    #     img_cut = cow_body_img[int(y0):int(y1), int(x0):int(x1), :]
    #     img_cut_name = img_path.split('/')[-1]
    #     print("img_cut_name", img_cut_name.replace('.jpg', '_'+str(i)+'.jpg', 1))
    #     cv2.imwrite(os.path.join(save_dir, img_class, img_cut_name.replace('.jpg', '_'+str(i)+'.jpg', 1)), img_cut)

    ### cow body
    for i, img_path in enumerate(all_img_list):
        if not img_path.endswith('.jpg'):
            continue
        data = detect_classfy_move_pic(aimodel, img_path)
        # print('data', data)
        if len(data)==0:
            continue
        if len(data) > 1:
            data = sorted(data, key=lambda x: x['box'][-1], reverse=True)
        pred_label = data[0]['label']
        if pred_label == 'blurry' or pred_label == '':
            continue
        # cow body copy
        if not os.path.exists(os.path.join(save_dir,'cow_body',pred_label)):
            os.makedirs(os.path.join(save_dir,'cow_body',pred_label))
        if i:
            print(i,'cow_body: ', img_path)
        shutil.copy(img_path, os.path.join(save_dir,'cow_body', pred_label))

        # bodai copy
        for j, single_box in enumerate(data):
            if single_box['label'] == "" or single_box['label'] == "blurry":
                continue
            pred_label = single_box['label']
            x0, y0, x1, y1,_ = single_box['box']
            if not os.path.exists(os.path.join(save_dir,'bodai', pred_label)):
                os.makedirs(os.path.join(save_dir,'bodai',pred_label))
            cow_body_img = cv2.imread(img_path)
            img_cut = cow_body_img[int(y0):int(y1), int(x0):int(x1), :]
            img_cut_name = img_path.split('/')[-1]
            if i:
                print(i,'bodai: ', os.path.join(save_dir,'bodai',pred_label, img_cut_name.replace('.jpg', '_' + str(i) + '.jpg', 1)))
            cv2.imwrite(os.path.join(save_dir,'bodai',pred_label, img_cut_name.replace('.jpg', '_' + str(i) + '.jpg', 1)),img_cut)






        # move_img_list.append([pred_label, img_path])
        # if len(move_img_list) % 100 == 0:
        #     print(len(move_img_list), 'of', i)

    ### copy cow body
    # for i, move_img in enumerate(move_img_list):
    #     img_class = move_img_list[i][0]
    #     img_path = move_img_list[i][1]
    #     if not os.path.exists(os.path.join(save_dir,img_class)):
    #         os.makedirs(os.path.join(save_dir,img_class))
    #     shutil.copy(img_path, os.path.join(save_dir, img_class))
    # return


if __name__ == '__main__':
    aimodel = Ai_model_three(config_path)

    ori_dir = r'/home/liusheng/data/NFS120/cow/vedio/16-202103/2-1-2021-03-17/Native Video Files (MP4)/images'
    save_dir = r'/home/liusheng/data/NFS120/cow/vedio/16-202103/2-1-2021-03-17/select_upload_data'

    cropPic(aimodel, ori_dir, save_dir, 1)





