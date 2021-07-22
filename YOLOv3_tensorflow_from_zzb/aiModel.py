"""
AI model infer
"""
import time
import cv2
import os
import configparser
# from utils import log
import numpy as np
import copy
# detection model
import tensorflow as tf
from model import yolov3
from model.utils import gpu_nms
import yaml

# classfy
import torch
# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from seresnet import se_resnet50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_class_names(class_name_path):
    names_list = []
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names_list.append(name.strip('\n'))
    return names_list


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


# detection model
class Detect_model:
    def __init__(self, model_path, num_class_path, input_size, anchor_path, threshold, iou_threshold):
        self.detect_obj_list = read_class_names(num_class_path)
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        with self.graph.as_default():
            anchors = parse_anchors(anchor_path)
            self.input_size = input_size
            self.input_data = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3], name='input_data')
            yolo_model = yolov3(len(self.detect_obj_list), anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            self.pred_boxes, self.pred_confs, self.pred_probs = yolo_model.predict(pred_feature_maps)

            self.pred_scores = self.pred_confs * self.pred_probs

            self.boxes, self.scores, self.labels = gpu_nms(self.pred_boxes, self.pred_scores, len(self.detect_obj_list), max_boxes=150,
                                                           score_thresh=threshold,
                                                           nms_thresh=iou_threshold)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)),
                               graph=self.graph)

        with self.graph.as_default():
            self.saver = tf.train.Saver()
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model_path)

    def forward(self, img):
        height_ori, width_ori = img.shape[:2]
        img = cv2.resize(img, tuple(self.input_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        boxes_, scores_, labels_, pred_confs_, pred_probs_ =self.sess.run(
            [self.boxes, self.scores, self.labels, self.pred_confs, self.pred_probs],
            feed_dict={self.input_data: img})
        boxes_[:, 0] *= (width_ori / float(self.input_size[0]))
        boxes_[:, 2] *= (width_ori / float(self.input_size[0]))
        boxes_[:, 1] *= (height_ori / float(self.input_size[1]))
        boxes_[:, 3] *= (height_ori / float(self.input_size[1]))
        return boxes_, scores_, labels_, pred_confs_, pred_probs_


class Classfy_model:
    def __init__(self, model_path, class_path, input_size, threshold=0.5):
        self.obj_list = read_class_names(class_path)

        self.classfy_model = se_resnet50(pretrained=None)
        if input_size[1] == 128:
            self.classfy_model.avg_pool = nn.AvgPool2d(4, stride=1)
        elif input_size[1] == 320:
            self.classfy_model.avg_pool = nn.AvgPool2d(10, stride=1)

        self.classfy_model.last_linear = nn.Linear(2048, len(self.obj_list))
        self.classfy_model.load_state_dict(torch.load(model_path))
        self.classfy_model = self.classfy_model.cuda()
        self.classfy_model.eval()
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.input_size = input_size  # [3, 320, 320]
        self.threshold = threshold

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        shift = [(w - nw) // 2, (h - nh) // 2]
        return new_image, scale, shift
    def forward(self, img):
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
        new_img, _, _ = self.letterbox_image(im, self.input_size[1:])
        # add
        input_data = self.test_transform(new_img)
        input_data = input_data.unsqueeze(0).to(device)
        output = self.classfy_model(input_data)
        output = F.softmax(output, dim=1)
        index = output.cpu().data.numpy().argmax()
        score = float(output.cpu().data.numpy().max())
        label = self.obj_list[index]
        return label, score

class Classfy_model_multil:
    def __init__(self, model_path, class_path, input_size, threshold=0.5):
        self.obj_list = read_class_names(class_path)

        self.classfy_model = se_resnet50(pretrained=None)
        if input_size[1] == 320:
            self.classfy_model.avg_pool = nn.AvgPool2d(4, stride=1)
            self.classfy_model.last_linear = nn.Linear(2048, len(self.obj_list))
        elif input_size[1] == 128:
            self.classfy_model.avg_pool = nn.AvgPool2d(10, stride=1)
            self.classfy_model.last_linear = nn.Linear(2048, len(self.obj_list))

        self.classfy_model.load_state_dict(torch.load(model_path))
        self.classfy_model = self.classfy_model.cuda()
        self.classfy_model.eval()
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.input_size = input_size  # [3, 320, 320]
        self.threshold = threshold

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        shift = [(w - nw) // 2, (h - nh) // 2]
        return new_image, scale, shift

    def forward(self, img):
        Sigmoid_fun = nn.Sigmoid()
        im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
        new_img, _, _ = self.letterbox_image(im, self.input_size[1:])
        # add
        input_data = self.test_transform(new_img)
        input_data = input_data.unsqueeze(0).to(device)
        output = self.classfy_model(input_data)
        output = Sigmoid_fun(output)
        label = output.cpu().data.numpy()[0].tolist()
        out_label = []
        for i, score in enumerate(label):
            if score > self.threshold:
                out_label.append(1)
            else:
                out_label.append(0)
        return label


class Ai_model:
    def __init__(self, config_path):
        with open(config_path, 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        print(config)
        # init classfy model
        classfy_model_path = config['classfy_model_path']
        classfy_class_num_path = config['classfy_class_num_path']
        classfy_input_size = config['classfy_input_size']
        classfy_th = config['classfy_th']
        self.classfy_model = Classfy_model(classfy_model_path, classfy_class_num_path, classfy_input_size, classfy_th)

        # init detect model
        detect_model_path = config['detect_model_path']
        detect_class_num_path = config['detect_class_num_path']
        detect_input_size = config['detect_input_size']
        detect_anchor_path = config['detect_anchor_path']
        detect_th = config['detect_th']
        detect_iou_th = config['detect_iou_th']
        self.detect_model = Detect_model(detect_model_path, detect_class_num_path, detect_input_size,
                                         detect_anchor_path, detect_th, detect_iou_th)

    def forward(self, img_path, is_show=True):
        # data = {"1": one_obj_data, "2": one_obj_data, "3": one_obj_data}
        data = []
        img = cv2.imread(img_path)
        img_ori = copy.deepcopy(img)
        img_show = copy.deepcopy(img)
        # detect
        boxes_, scores_, labels_, pred_confs_, pred_probs_ = self.detect_model.forward(img)

        # classfy
        for i in range(len(boxes_)):
            one_obj_data = {"label": "", "color": "", "line_mode": "", "line_color": "", "box": []}
            x0, y0, x1, y1 = boxes_[i]
            s = (x1 - x0) * (y1 - y0)
            x0 = max(0, x0)
            x1 = max(0, x1)
            y0 = max(0, y0)
            y1 = max(0, y1)
            if s > 0:
                img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
                obj, classfy_score = self.classfy_model.forward(img_cut)
                # label
                one_obj_data["label"] = obj
                # box
                one_obj_data["box"] = [aa for aa in [int(x0), int(y0), int(x1), int(y1), float(scores_[i]), float(classfy_score)]]
                # data
                data.append(one_obj_data)
                if is_show:
                    cv2.rectangle(img_show, (int(x0), int(y0)),
                                  (int(x1), int(y1)),
                                  [255, 255, 0], 5)
                    cv2.putText(img_show, obj, (int(x0), int(y0) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

        return data, img_show


class Ai_model_three:
    def __init__(self, config_path):
        with open(config_path, 'r') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        print(config)
        # init classfy model
        classfy_model_path = config['classfy_model_path']
        classfy_class_num_path = config['classfy_class_num_path']
        classfy_input_size = config['classfy_input_size']
        classfy_th = config['classfy_th']
        self.classfy_model = Classfy_model(classfy_model_path, classfy_class_num_path, classfy_input_size, classfy_th)

        # init classfy model
        classfy_cb_model_path = config['classfy_cb_model_path']
        classfy_cb_class_num_path = config['classfy_cb_class_num_path']
        classfy_cb_input_size = config['classfy_cb_input_size']
        classfy_cb_th = config['classfy_cb_th']
        self.classfy_cb_model = Classfy_model(classfy_cb_model_path, classfy_cb_class_num_path, classfy_cb_input_size, classfy_cb_th)

        # init detect model
        detect_model_path = config['detect_model_path']
        detect_class_num_path = config['detect_class_num_path']
        detect_input_size = config['detect_input_size']
        detect_anchor_path = config['detect_anchor_path']
        detect_th = config['detect_th']
        detect_iou_th = config['detect_iou_th']
        self.detect_model = Detect_model(detect_model_path, detect_class_num_path, detect_input_size,
                                         detect_anchor_path, detect_th, detect_iou_th)

    # def forward(self, img_path, is_show=True):
    #     # data = {"1": one_obj_data, "2": one_obj_data, "3": one_obj_data}
    #     data = []
    #     img = cv2.imread(img_path)
    #     img_ori = copy.deepcopy(img)
    #     img_show = copy.deepcopy(img)
    #     # detect
    #     boxes_, scores_, labels_, pred_confs_, pred_probs_ = self.detect_model.forward(img)
    #
    #     # classfy
    #     for i in range(len(boxes_)):
    #         one_obj_data = {"label": "", "color": "", "line_mode": "", "line_color": "", "box": []}
    #         x0, y0, x1, y1 = boxes_[i]
    #         s = (x1 - x0) * (y1 - y0)
    #         x0 = max(0, x0)
    #         x1 = max(0, x1)
    #         y0 = max(0, y0)
    #         y1 = max(0, y1)
    #         if s > 0:
    #             img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
    #             # 判断图像是清晰还是模糊
    #             obj, classfy_score = self.classfy_cb_model.forward(img_cut)
    #
    #             if obj == 'clear' and classfy_score > self.classfy_cb_model.threshold:
    #                 obj, classfy_score = self.classfy_model.forward(img_cut)
    #
    #             # label
    #             one_obj_data["label"] = obj
    #             # box
    #             one_obj_data["box"] = [aa for aa in [int(x0), int(y0), int(x1), int(y1), float(scores_[i]), float(classfy_score)]]
    #             # data
    #             data.append(one_obj_data)
    #             if is_show:
    #                 cv2.rectangle(img_show, (int(x0), int(y0)),
    #                               (int(x1), int(y1)),
    #                               [255, 255, 0], 5)
    #                 cv2.putText(img_show, str(scores_[i]), (int(x0), int(y0) + 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
    #
    #     return data, img_show
    def forward(self, img_path):
        img = cv2.imread(img_path)
        img_ori = copy.deepcopy(img)
        boxes_, scores_, labels_, pred_confs_, pred_probs_ = self.detect_model.forward(img_ori)
        data=[]
        # classfy
        # print("boxes_", len(boxes_))
        for i in range(len(boxes_)):
            one_obj_data = {"label": "", "box": []}
            x0, y0, x1, y1 = boxes_[i]
            s = (x1 - x0) * (y1 - y0)
            x0 = max(0, x0)
            x1 = max(0, x1)
            y0 = max(0, y0)
            y1 = max(0, y1)
            if s > 0:
                img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
                obj, classfy_score = self.classfy_cb_model.forward(img_cut)
                # print("1", obj, classfy_score)

                if obj == 'clear' and classfy_score > self.classfy_cb_model.threshold:
                    obj, classfy_score = self.classfy_model.forward(img_cut)

                ### 可选值为 "blurry","clear"+ bodai class, ### classfy_score, clear -> cb score, class name: conf
                # one_obj_data["label"] = obj
                # one_obj_data["box"] = [aa for aa in
                #                        [int(x0), int(y0), int(x1), int(y1), float(scores_[i]), float(classfy_score)]]
                if classfy_score > 0.85:
                    one_obj_data["label"] = obj
                    one_obj_data["box"] = [int(x0), int(y0), int(x1), int(y1), float(classfy_score)]
                    data.append(one_obj_data)
        return data

