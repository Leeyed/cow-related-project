# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np

from model.utils import parse_anchors
from model.utils import gpu_nms
from model import yolov3, yolov3_bodai
import cv2

# detection model for bodai
class bodai_Model:
    def __init__(self, model_path, num_class, input_size, anchors_path, threshold, iou_threshold):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        with self.graph.as_default():
            anchors = parse_anchors(anchors_path)
            self.input_size = input_size
            self.input_data = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3], name='input_data')
            yolo_model = yolov3_bodai(num_class, anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            self.pred_boxes, self.pred_confs, self.pred_probs = yolo_model.predict(pred_feature_maps)

            self.pred_scores = self.pred_confs * self.pred_probs

            self.boxes, self.scores, self.labels = gpu_nms(self.pred_boxes, self.pred_scores, num_class, max_boxes=150, score_thresh=threshold,
                                            nms_thresh=iou_threshold)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)), graph=self.graph)

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