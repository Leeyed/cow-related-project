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

from networks.aiModel import detectModel, registerModel
from networks.u2net import segmentModel

from modules_utils.tools import normPRED, save_output_mask_rect, save_infer_feature, readMaskData, boxValid
from modules_utils.dataset import SalObjDataset, RescaleT, ToTensorLab


def video_process(config: dict, detect_model: detectModel, segment_model: segmentModel, register_model: registerModel):
    url = r'/home/liusheng/data/NFS120/cow/vedio/15组2021.5/DVR_Examiner_Export_2021-05-18 142552_Job_0002/2021-05-12/Native Video Files (MP4)/fade'
    save_url = r'/home/liusheng/data/NFS120/cow/vedio/15组2021.5/DVR_Examiner_Export_2021-05-18 142552_Job_0002/2021-05-12/Native Video Files (MP4)/save'
    # save_url = r''
    for folder in os.listdir(url):
        for img_path in os.listdir(os.path.join(url, folder)):
            img_abs_path = os.path.join(url, folder, img_path)
            img = cv2.imread(img_abs_path)
            print(f'img_path:{img_abs_path}')
            pred = segment_model.forward(img)

            _ = os.path.join(save_url, folder)
            if not os.path.exists(_):
                os.makedirs(_)
            cv2.imwrite(os.path.join(_, img_path), pred)
            # _ = os.path.join(save_url, folder)
            # if not os.path.exists(_):
            #     os.makedirs(_)
            # save_output_mask_rect(img_abs_path, pred, _)




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