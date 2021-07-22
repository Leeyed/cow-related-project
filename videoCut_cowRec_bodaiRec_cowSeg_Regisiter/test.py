"""
this s
"""

import yaml
import torch
import os
import cv2
import copy
import datetime
import pickle
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from skimage import io, transform
from skimage import img_as_float

from networks.aiModel import detectModel, registerModel
from networks.u2net import segmentModel

from modules_utils.tools import normPRED, save_output_mask_rect, save_infer_feature, readMaskData, boxValid
from modules_utils.dataset import SalObjDataset, RescaleT, ToTensorLab


def video_process(config: dict, detect_model: detectModel, segment_model: segmentModel, register_model: registerModel):
    video_path = config['video_path']
    video_list = sorted(os.listdir(video_path))
    video_list = list(filter(lambda x: x.endswith(('.mp4', '.MP4')), video_list))
    print(len(video_list))
    mask_data = readMaskData('mask_image_data')
    for video in video_list:
        processed_detection = not config['detection_flag']
        processed_mask125_sameway = not config['mask125_sameway_flag']
        processed_registration = not config['registration_flag']

        # detect
        video_imgs_path = os.path.join(video_path, "images_all_test", video[:-4])
        if not os.path.exists(video_imgs_path):
            os.makedirs(video_imgs_path)
        else:
            print(f"{video} had save pics!")
            processed_detection = True
        if not processed_detection:
            cap = cv2.VideoCapture(os.path.join(video_path, video))
            num = 0
            while True:
                num += 1
                ret, frame = cap.read()
                if not ret:
                    print(f"{video} video is over!")
                    break
                if num % 105 != 0: continue
                img_ori = copy.deepcopy(frame)
                boxes, scores, labels = detect_model.forword(img_ori)
                for i in range(len(boxes)):
                    x0, y0, x1, y1 = boxes[i]
                    s = (x1 - x0) * (y1 - y0)
                    if s < 0: continue
                    x0, x1, y0, y1 = max(0, x0), max(0, x1), max(0, y0), max(0, y1)
                    x0, x1, y0, y1 = min(x0, img_ori.shape[1]), min(x1, img_ori.shape[1]), min(y0, img_ori.shape[0]), min(
                        y1, img_ori.shape[0])
                    center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
                    if not boxValid(mask_data, center, video, img_ori.shape[:2]): continue
                    img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
                    time_str = datetime.datetime.now().strftime('_%H%M%S_%f')
                    img_cut_name = video[:-4] + time_str + '.jpg'
                    try:
                        cv2.imwrite(os.path.join(video_imgs_path, img_cut_name), img_cut)
                        print(img_cut_name, scores[i])
                    except:
                        print("img_cut is None!")
            cap.release()
        # one video complete

        # draw masked_125 & sameway images
        mask125_save_dir = os.path.join(video_path, "mask125_images_test", video[:-4])
        if not os.path.exists(mask125_save_dir):
            os.makedirs(mask125_save_dir)
        else:
            print(f"{video} had save masked-125 sameway pics!")
            processed_mask125_sameway = True
        if 1:
            imgs = os.listdir(video_imgs_path)
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            imgs = [os.path.join(video_imgs_path, img) for img in imgs]
            print("imgs num: ", len(imgs))
            # test_salobj_dataset = SalObjDataset(img_name_list=imgs,
            #                                     lbl_name_list=[],
            #                                     transform=transforms.Compose([RescaleT(320),  # 480 == 线缝   320 == 牛体
            #                                                                   ToTensorLab(flag=0)])
            #                                     )
            # test_salobj_dataloader = DataLoader(test_salobj_dataset,
            #                                     batch_size=1,
            #                                     shuffle=False,
            #                                     num_workers=4)
            # for i, data_test in enumerate(test_salobj_dataloader):
            #     print("inferencing:", imgs[i])
            #     inputs_test = data_test['image']
            #     inputs_test = inputs_test.type(torch.FloatTensor)
            #     if torch.cuda.is_available():
            #         inputs_test = Variable(inputs_test.cuda())
            #     else:
            #         inputs_test = Variable(inputs_test)
            #     print(inputs_test)
            #     print(inputs_test.size())
            #
            #     d1, d2, d3, d4, d5, d6, d7 = segment_model.forward(inputs_test)
            #     #
            #     # normalization
            #     pred = d1[:, 0, :, :]
            #     pred = normPRED(pred)
            #     if not os.path.exists(mask125_save_dir):
            #         os.makedirs(mask125_save_dir)
            #     save_output_mask_rect(imgs[i], pred, mask125_save_dir)
            #     del d1, d2, d3, d4, d5, d6, d7
            #     exit()

            for i, img in enumerate(imgs):
                print("inferencing:", imgs[i])

                # function 1
                # im = io.imread(img)
                # input_img = transform.resize(im, (320,320), mode='constant')
                # input_img = input_img.transpose(2,0,1)
                # input_img = torch.FloatTensor(input_img)
                # input_img = input_img.unsqueeze(0).to(config['device'])

                # function2
                im = cv2.imread(img)
                # image = img_as_float(im)
                # input_img = transform.resize(image, (320,320), mode='constant')
                # input_img = input_img.transpose(2,0,1)
                # input_img = torch.FloatTensor(input_img)
                # input_img = input_img.unsqueeze(0).to(config['device'])

                # d1, d2, d3, d4, d5, d6, d7 = segment_model.forward(input_img)
                pred = segment_model.forward(im)
                # normalization
                # pred = d1[:, 0, :, :]
                # pred = normPRED(pred)
                if not os.path.exists(mask125_save_dir):
                    os.makedirs(mask125_save_dir)
                save_output_mask_rect(imgs[i], pred, mask125_save_dir)
                exit()

        # register
        pkl_dir = os.path.join(config['video_path'], 'pickles_test', video[:-4])
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        else:
            print(f"{video} had save features!")
            processed_registration = True
        if not processed_registration:
            save_infer_feature(video_imgs_path, os.path.join(pkl_dir, 'feature.pkl'), register_model, config)
        # f = open(pkl_path, 'rb')
        # feature_dict = pickle.load(f)
        # f.close()
        # for key, value in feature_dict.items():
        #     feature_dict[key] = torch.from_numpy(value).to(config['device'])
        break

if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    for key, val in config.items():
        print(key, val)

    detect_model = None
    segment_model = segmentModel(config)
    register_model = None

    video_process(config, detect_model, segment_model, register_model)