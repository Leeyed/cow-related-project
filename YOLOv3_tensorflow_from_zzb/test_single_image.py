# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

from model.utils import parse_anchors, read_class_names
from model.utils import gpu_nms
from model.utils import get_color_table, plot_one_box
from model.utils import letterbox_resize
import random
from model import yolov3

import args

# parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
# parser.add_argument("input_image", type=str,
#                     help="The path of the input image.")
# parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
#                     help="The path of the anchor txt file.")
# parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
#                     help="Resize the input image with `new_size`, size format: [width, height]")
# parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
#                     help="Whether to use the letterbox resize.")
# parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
#                     help="The path of the class names.")
# parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
#                     help="The path of the weights to restore.")
# args = parser.parse_args()

args.restore_path = r'/home/liusheng/deepLearningProjects/projects/YOLOv3_tensorflow_from_zzb/checkpoint/Cow_qcj_0831_model-epoch_98_step_18215_loss_0.3877_lr_4.6042e-05_one'
args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

# args.input_image = '111.jpg'
args.new_size = [512, 512]
with open('data/my_data/test.txt', 'r') as f:
    data = f.readlines()

images = []
for d in data:
    img_url = d.strip().split(' ')[1]
    # img_url = img_url.replace('/home/liusheng/data/NFS120', r'X:')
    print(img_url)
    images.append(img_url)
    # exit()

for input_image in images:
    tf.reset_default_graph()
    img_ori = cv2.imread(input_image)
    # if not img_ori: continue
    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=100, score_thresh=0.1,
                                        nms_thresh=0.05)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1],
                         label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                         color=color_table[labels_[i]])
    img_name = random.randint(0,100000)
    cv2.imwrite(os.path.join('/home/liusheng/data/NFS120/barcode/ZVZ-real-512/img', str(img_name)+'.jpg'), img_ori)
    # cv2.imshow('Detection result', img_ori)
        # cv2.imwrite('detection_result.jpg', img_ori)
    # cv2.waitKey(0)
        # exit()
