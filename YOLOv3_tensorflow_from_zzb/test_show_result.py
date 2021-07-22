# coding: utf-8

from __future__ import division, print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'


import tensorflow as tf
import numpy as np
import argparse
from tqdm import trange

from model.utils import get_batch_data
from model.utils import parse_anchors, read_class_names, AverageMeter
from model.utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from model.utils import gpu_nms
from model.utils import get_color_table, plot_one_box

from model import yolov3
import cv2
import time
import os
import copy
import datetime
from bodai_detect import bodai_Model



parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str, default="/home/zhouzhubin/workspace/project/datasets/cow/JPEGImages/ch04_20190827165732_000654_00000001.png",
                    help="The path of the input image.")
# parser.add_argument("--anchor_path", type=str, default="./data/my_data/cotton_anchors.txt",
#                     help="The path of the anchor txt file.")
parser.add_argument("--anchor_path", type=str, default="./data/my_data/cow_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],  #[5440, 5440],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/cow_big_class.txt",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/Cow_qcj_0831_model-epoch_99_step_23699_loss_18.8626_lr_4.6042e-05_one",
                    help="The path of the weights to restore.")

args, unknown = parser.parse_known_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)
color_table = get_color_table(args.num_class)


def find_bbox_label(bb_label_data):
    bboxs = []
    labels = []
    data_len = len(bb_label_data)
    for i in range(data_len//5):  # label bbox
        bboxs.append((float(bb_label_data[i * 5 + 1]),
                      float(bb_label_data[i * 5 + 2]),
                      float(bb_label_data[i * 5 + 3]),
                      float(bb_label_data[i * 5 + 4])))
        labels.append(bb_label_data[i * 5])
    return bboxs, labels


def inference_img_dir():
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        # pred_scores = pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=150, score_thresh=0.2,
                                        nms_thresh=0.2)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        # txt file
        # txt_data = './data/my_data/bcs_20200429.txt'
        # for line in open(txt_data, 'r').readlines():
        #     s = line.strip().split(' ')
        #     input_image = s[0]
        #     frame = cv2.imread(os.path.join('/home/zhouzhubin/sjht_data/', input_image))

        # img_dir
        # img_dir = '/home/zhouzhubin/workspace/project/datasets/cow/JPEGImages/'
        # img_dir = '/home/zhouzhubin/data/cowrecognition/monitor_video/pickdir/0428/042803'
        img_dir = '/home/zhouzhubin/sjht_data/images/gmnnc-bs/gmnnc-bs-2020051204/1'
        # img_dir = '/home/zhouzhubin/NFS_AIDATA/bodyscore/BCS/GH010131'
        # img_dir = '/home/zhouzhubin/NFS_AIDATA/bodyscore/0422/images'

        img_list = os.listdir(img_dir)
        for input_image in img_list:

            # frame = cv2.imread(args.input_image)
            frame = cv2.imread(os.path.join(img_dir, input_image))
            print(input_image)

            start = time.time()

            img_ori = frame
            height_ori, width_ori = img_ori.shape[:2]

            img = cv2.resize(img_ori, tuple(args.new_size))
            img_show = copy.deepcopy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            boxes_, scores_, labels_, pred_confs_, pred_probs_ = sess.run(
                [boxes, scores, labels, pred_confs, pred_probs],
                feed_dict={input_data: img})
            #
            # for i in range(len(boxes_)):
            #     x0, y0, x1, y1 = boxes_[i]
            #     s = (x1 - x0) * (y1 - y0)
            #     if s > 0:
            #         plot_one_box(img_show, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]], conf=scores_[i])

            # rescale the coordinates to the original image
            boxes_[:, 0] *= (width_ori / float(args.new_size[0]))
            boxes_[:, 2] *= (width_ori / float(args.new_size[0]))
            boxes_[:, 1] *= (height_ori / float(args.new_size[1]))
            boxes_[:, 3] *= (height_ori / float(args.new_size[1]))

            # print("box coords:")
            # print(boxes_)
            # print('*' * 30)
            # print("scores:")
            # print(scores_)
            # print('*' * 30)
            # print("labels:")
            # print(labels_)

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                s = (x1 - x0) * (y1 - y0)
                if s > 0:
                    plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]],
                                 color=color_table[labels_[i]], conf=scores_[i])

            # cv2.imwrite('find.jpg', img_ori)
            # img_ori = cv2.resize(img_ori, (960, 960))
            cv2.namedWindow("Detection result", cv2.WND_PROP_FULLSCREEN)
            end = time.time()
            cv2.imshow('Detection result', img_ori)
            # cv2.imshow('show', img_show)
            print("time is %f, number is %d, hair rata is %f, not hair rata is %f" % (
                (end - start), len(boxes_), len(boxes_) / 36, len(boxes_) / 31))
            # cv2.imwrite('detection_result.jpg', img_ori)
            cv2.waitKey(0)


def inference_txt(txt):
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        # pred_scores = pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=150, score_thresh=0.8,
                                        nms_thresh=0.5)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        # txt file
        # txt_data = './data/my_data/bcs_20200429.txt'
        # for line in open(txt_data, 'r').readlines():
        #     s = line.strip().split(' ')
        #     input_image = s[0]
        #     frame = cv2.imread(os.path.join('/home/zhouzhubin/sjht_data/', input_image))

        # img_dir

        img_dir = '/home/zhouzhubin/sjht_data/'

        data = open(txt, 'r')
        line_datas = data.readlines()
        for line_data in line_datas:

            line_list = line_data.strip('\n').split(' ')
            line_list = line_list[1:]  # 去除id
            input_image = line_list[0]

            frame = cv2.imread(os.path.join(img_dir, input_image))
            print(input_image)

            start = time.time()

            img_ori = frame
            height_ori, width_ori = img_ori.shape[:2]

            img = cv2.resize(img_ori, tuple(args.new_size))
            img_show = copy.deepcopy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            boxes_, scores_, labels_, pred_confs_, pred_probs_ = sess.run(
                [boxes, scores, labels, pred_confs, pred_probs],
                feed_dict={input_data: img})
            #
            # for i in range(len(boxes_)):
            #     x0, y0, x1, y1 = boxes_[i]
            #     s = (x1 - x0) * (y1 - y0)
            #     if s > 0:
            #         plot_one_box(img_show, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]], conf=scores_[i])

            # rescale the coordinates to the original image
            boxes_[:, 0] *= (width_ori / float(args.new_size[0]))
            boxes_[:, 2] *= (width_ori / float(args.new_size[0]))
            boxes_[:, 1] *= (height_ori / float(args.new_size[1]))
            boxes_[:, 3] *= (height_ori / float(args.new_size[1]))

            bb_label_data = line_list[1:]
            bboxs, labels_GT = find_bbox_label(bb_label_data)
            num = 0
            for bbox in bboxs:
                cv2.rectangle(img_ori, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [255, 255, 0], 5)
                if labels_GT[num].startswith('b'):
                    cv2.putText(img_ori, str(labels_GT[num]), (int(bbox[0]), int(bbox[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
                elif labels_GT[num].startswith('n'):
                    cv2.putText(img_ori, str(labels_GT[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
                elif labels_GT[num].startswith('c'):
                    cv2.putText(img_ori, str(labels_GT[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
                num += 1

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                s = (x1 - x0) * (y1 - y0)
                if s > 0:
                    plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]],
                                 color=color_table[labels_[i]], conf=scores_[i])

            # cv2.imwrite('find.jpg', img_ori)
            # img_ori = cv2.resize(img_ori, (960, 960))
            cv2.namedWindow("Detection result", cv2.WND_PROP_FULLSCREEN)
            end = time.time()
            cv2.imshow('Detection result', img_ori)
            # cv2.imshow('show', img_show)
            print("time is %f, number is %d, hair rata is %f, not hair rata is %f" % (
                (end - start), len(boxes_), len(boxes_) / 36, len(boxes_) / 31))
            # cv2.imwrite('detection_result.jpg', img_ori)
            cv2.waitKey(0)


def cut_pic_from_videos():
    anchor_path = './data/bodai/bodai_anchors.txt'
    input_size = [416, 416]
    model_path = './checkpoint/bodai_0919_model-epoch_99_step_44699_loss_1.8250_lr_4.6042e-05'

    bodai_model = bodai_Model(model_path, 1, input_size, anchor_path, 0.2, 0.2)


    print("Start!!!")
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        # pred_scores = pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=150, score_thresh=0.8,
                                        nms_thresh=0.1)

        saver = tf.train.Saver()
        print("start restore")
        saver.restore(sess, args.restore_path)
        print("end restore")

        video_path = '/home/zhouzhubin/NFS_AIDATA/cow_video/cowid/suqian-15-1115/DVR_Examiner_Export_2020-11-23 155733_Job_0002/2020-11-15/Native Video Files (MP4)/'
        video_list = sorted(os.listdir(video_path))
        print(len(video_list))
        for video in video_list:

            if video.endswith('.mp4') or video.endswith('.MP4'):
                video_imgs_path = os.path.join(video_path, "images", video[:-4])
                if not os.path.exists(video_imgs_path):
                    os.makedirs(video_imgs_path)
                else:
                    print("This video had save pics!")
                    continue

                cap = cv2.VideoCapture(os.path.join(video_path, video))

                fps = 0.0
                num = 0
                while True:
                    t1 = time.time()
                    ret, frame = cap.read()

                    if not ret:
                        print("video is over!")
                        break
                    if num == 105:
                        num = 0
                        img_ori = copy.deepcopy(frame)
                        height_ori, width_ori = img_ori.shape[:2]
                        img = cv2.resize(frame, tuple(args.new_size))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = np.asarray(img, np.float32)
                        img = img[np.newaxis, :] / 255.
                        boxes_, scores_, labels_, pred_confs_, pred_probs_ = sess.run(
                            [boxes, scores, labels, pred_confs, pred_probs],
                            feed_dict={input_data: img})
                        # box转化
                        boxes_[:, 0] *= (width_ori / float(args.new_size[0]))
                        boxes_[:, 2] *= (width_ori / float(args.new_size[0]))
                        boxes_[:, 1] *= (height_ori / float(args.new_size[1]))
                        boxes_[:, 3] *= (height_ori / float(args.new_size[1]))
                        for i in range(len(boxes_)):
                            x0, y0, x1, y1 = boxes_[i]
                            s = (x1 - x0) * (y1 - y0)
                            x0 = max(0, x0)
                            x1 = max(0, x1)
                            y0 = max(0, y0)
                            y1 = max(0, y1)
                            if s > 0:
                                # plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]],
                                #              color=color_table[labels_[i]], conf=scores_[i])
                                time_str = datetime.datetime.now().strftime('_%H%M%S_%f')
                                img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
                                boxes_bodai, _, _, _, _ = bodai_model.forward(img_cut)
                                if len(boxes_bodai) == 0:
                                    continue
                                #for box_b in boxes_bodai:
                                #    x0_b, y0_b, x1_b, y1_b = box_b
                                #    cv2.rectangle(img, (int(x0_b), int(y0_b)), (int(x1_b), int(y1_b)), [0, 0, 0], 4)
                                img_cut_name = video[:-4] + time_str + '.jpg'
                                print(img_cut_name)
                                try:
                                    
                                    cv2.imwrite(os.path.join(video_imgs_path, img_cut_name), img_cut)
                                except:
                                    print("img_cut is None!")
                        #cv2.namedWindow("Detection result", cv2.WND_PROP_FULLSCREEN)
                        #cv2.imshow('Detection result', img_ori)
                        #cv2.waitKey(0)
                    num += 1
                cap.release()


if __name__ == '__main__':
    # txt = './data/my_data/val_qcj_20200827_data_with_id.txt'
    # inference_txt(txt)
    cut_pic_from_videos()
