"""
AI model infer
"""
import cv2
import tensorflow as tf
from model import yolov3
from modules.nms_utils import gpu_nms
import torch.nn as nn
import numpy as np
import torch

from seresnet import se_resnet50
from modules_utils.seresnet import se_resnet50
from modules_utils.networks import resnetface20
from efficientnet_pytorch import EfficientNet


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


# cow body detection model
class detectModel:
    """
    detect model
    """
    def __init__(self, config):
        self.detect_obj_list = read_class_names(config['body_detect_class_path'])
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        # self.socre_th = config['body_detect_score_th']
        with self.graph.as_default():
            anchors = parse_anchors(config['body_detect_anchor_path'])
            self.input_size = config['body_detect_input_size']
            self.input_data = tf.placeholder(tf.float32, [1, self.input_size[0], self.input_size[1], 3], name='input_data')
            yolo_model = yolov3(len(self.detect_obj_list), anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            self.pred_boxes, self.pred_confs, self.pred_probs = yolo_model.predict(pred_feature_maps)
            self.pred_scores = self.pred_confs * self.pred_probs

            self.boxes, self.scores, self.labels = gpu_nms(self.pred_boxes, self.pred_scores,
                                                           len(self.detect_obj_list),
                                                           max_boxes=150,
                                                           score_thresh=config['body_detect_th'],
                                                           nms_thresh=config['body_detect_iou_th'])
        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)),
            graph=self.graph)

        with self.graph.as_default():
            self.saver = tf.train.Saver()
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, config['body_detect_model_path'])

    def forword(self, img):
        """
        input: opencv bgr image
        """
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
        # return boxes_, scores_, labels_, pred_confs_, pred_probs_
        # boxes_, scores_, labels_ = np.array(boxes_), np.array(scores_), np.array(labels_)
        # f_index = np.where(np.array(scores_>self.socre_th))
        # boxes_, scores_, labels_ = boxes_[f_index], scores_[f_index], labels_[f_index],
        return boxes_, scores_, labels_


class registerModel:
    def __init__(self, config:dict):
        global backbone
        if config['backbone'] == 'resnetface20':
            backbone = resnetface20(input_shape=config['input_shape'])
        elif config['backbone'] == 'se_resnet50':
            backbone = se_resnet50(pretrained='imagenet')
            if config['input_shape'][1] == 128:
                # 128 bodai
                backbone.avg_pool = nn.AvgPool2d(4, stride=1)
            elif config['input_shape'][1] == 320:
                # 320 cow body
                backbone.avg_pool = nn.AvgPool2d(10, stride=1)
                # 192 cow body register
            elif config['input_shape'][1] == 192:
                backbone.avg_pool = nn.AvgPool2d(6, stride=1)
            backbone.last_linear = nn.Linear(2048, 512)
        elif 'efficientnet' in config['backbone']:
            backbone = EfficientNet.from_pretrained(config['backbone'])
            feature = backbone._fc.in_features
            backbone._fc = nn.Linear(in_features=feature, out_features=512, bias=True)
        else:
            assert 'backbone error'

        model_path = config['register_model_path']
        print('load model', model_path)
        assert isinstance(backbone, object)
        backbone.load_state_dict(torch.load(model_path, map_location=config['device']))
        backbone.to(config['device'])
        backbone.eval()
        self.backbone = backbone

    def forward(self, batch_data):
        # input_data = prepare_image(image_path, (input_shape[2], input_shape[1]), transform)
        # input_data = input_data.unsqueeze(0).to(config['device'])
        # batch_keys.append(image_path)
        # batch_data.append(input_data)
        # batch_data = torch.cat(batch_data)
        return self.backbone(batch_data)
