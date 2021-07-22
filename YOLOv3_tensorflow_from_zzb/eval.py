# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
from tqdm import trange

from model.utils import get_batch_data
from model.utils import parse_anchors, read_class_names, AverageMeter
from model.utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec, voc_eval_for_confindence
from model.utils import gpu_nms

from model import yolov3


# add by zhouzhubin 用于根据得分阈值来,重新生成用于计算recall, presion的列表
def create_new_list(threshold, all_list): # (img_id, xmin, ymin, xmax, ymax, score, label)
    new_list = []
    for one in all_list:
        if one[5] >= threshold:
            new_list.append(one)
    return new_list


#################
# ArgumentParser
#################
parser = argparse.ArgumentParser(description="YOLO-V3 eval procedure.")
# some paths
parser.add_argument("--eval_file", type=str, default="./data/my_data/val_qcj_20200827_data_with_id.txt",
                    help="The path of the validation or test txt file.")

parser.add_argument("--restore_path", type=str, default="./checkpoint/Cow_qcj_0827_model-epoch_45_step_10901_loss_8.0711_lr_6.9253e-05_one",
                    help="The path of the weights to restore.")

parser.add_argument("--anchor_path", type=str, default="./data/my_data/cow_anchors.txt",
                    help="The path of the anchor txt file.")

parser.add_argument("--class_name_path", type=str, default="./data/my_data/cow_class.txt",
                    help="The path of the class names.")

# some numbers
parser.add_argument("--img_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image to `img_size`, size format: [width, height]")

parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to use the letterbox resize, i.e., keep the original image aspect ratio.")

parser.add_argument("--num_threads", type=int, default=10,
                    help="Number of threads for image processing used in tf.data pipeline.")

parser.add_argument("--prefetech_buffer", type=int, default=5,
                    help="Prefetech_buffer used in tf.data pipeline.")

parser.add_argument("--nms_threshold", type=float, default=0.75,
                    help="IOU threshold in nms operation.")

parser.add_argument("--score_threshold", type=float, default=0.0,
                    help="Threshold of the probability of the classes in nms operation.")

parser.add_argument("--nms_topk", type=int, default=150,
                    help="Keep at most nms_topk outputs after nms.")

parser.add_argument("--use_voc_07_metric", type=lambda x: (str(x).lower() == 'true'), default=False,
                    help="Whether to use the voc 2007 mAP metrics.")

args = parser.parse_args()

# args params
args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
# args.class_num = len(args.classes)
args.class_num = 1
args.img_cnt = len(open(args.eval_file, 'r').readlines())

# setting placeholders
is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

##################
# tf.data pipeline
##################
val_dataset = tf.data.TextLineDataset(args.eval_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data, [x, args.class_num, args.img_size, args.anchors, 'val', False, False, args.letterbox_resize], [tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)
iterator = val_dataset.make_one_shot_iterator()

image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
image_ids.set_shape([None])
y_true = [y_true_13, y_true_26, y_true_52]
image.set_shape([None, args.img_size[1], args.img_size[0], 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])

##################
# Model definition
##################
yolo_model = yolov3(args.class_num, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(image, is_training=is_training)
loss = yolo_model.compute_loss(pred_feature_maps, y_true)
y_pred = yolo_model.predict(pred_feature_maps)

saver_to_restore = tf.train.Saver()

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    saver_to_restore.restore(sess, args.restore_path)

    print('\n----------- start to eval -----------\n')

    val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
        AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    val_preds = []

    for j in trange(args.img_cnt):
        __image_ids, __y_pred, __loss = sess.run([image_ids, y_pred, loss], feed_dict={is_training: False})
        pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)

        val_preds.extend(pred_content)
        val_loss_total.update(__loss[0])
        val_loss_xy.update(__loss[1])
        val_loss_wh.update(__loss[2])
        val_loss_conf.update(__loss[3])
        val_loss_class.update(__loss[4])

    score_threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    for pre_t in score_threshold_list:
        print("当前阈值为: ", pre_t)
        # calc mAP
        rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
        gt_dict = parse_gt_rec(args.eval_file, args.img_size, args.letterbox_resize)

        info = '==========================\n'
        new_list = create_new_list(pre_t, val_preds)
        for ii in range(args.class_num):
            npos, nd, rec, prec, ap = voc_eval(gt_dict, new_list, ii, iou_thres=args.nms_threshold,
                                               use_07_metric=args.use_voc_07_metric)
            # info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
            info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}\n'.format(ii, rec, prec)
            rec_total.update(rec, npos)
            prec_total.update(prec, nd)
            ap_total.update(ap, 1)

        mAP = ap_total.average

        info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}\n'.format(rec_total.average,
                                                                  prec_total.average)
        # 对所有的符合IOU的检测框做recall以及precision
        npos, nd, rec, prec, ap = voc_eval_for_confindence(gt_dict, new_list, args.class_num,
                                                           iou_thres=args.nms_threshold,
                                                           use_07_metric=args.use_voc_07_metric)
        info += 'Condience EVAL: Recall: {:.4f}, Precison: {:.4f}\n'.format(rec,
                                                                            prec)
        info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
            val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average,
            val_loss_class.average)

        print(info)
    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    # old 只是用来做记录 # 计算总的多个类别的以及confidence的map
    info_ = "多个类别的ap, map\n"
    for ii in range(args.class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=args.nms_threshold,
                                           use_07_metric=args.use_voc_07_metric)
        info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
        info_ += 'EVAL: Class {}: AP: {:.4f}\n'.format(ii, ap)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)

    mAP = ap_total.average
    info_ += 'EVAL: mAP: {:.4f}\n'.format(mAP)
    info_ += "confidence map\n"
    npos, nd, rec, prec, ap = voc_eval_for_confindence(gt_dict, val_preds, args.class_num,
                                                       iou_thres=args.nms_threshold,
                                                       use_07_metric=args.use_voc_07_metric)
    info_ += 'Condience EVAL: mAP: {:.4f}\n'.format(ap)
    print(info_)

    # rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    # gt_dict = parse_gt_rec(args.eval_file, args.img_size, args.letterbox_resize)
    # print('mAP eval:')
    # for ii in range(args.class_num):
    #     npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=0.5, use_07_metric=args.use_voc_07_metric)
    #     rec_total.update(rec, npos)
    #     prec_total.update(prec, nd)
    #     ap_total.update(ap, 1)
    #     print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(ii, rec, prec, ap))
    #
    # mAP = ap_total.average
    # print('final mAP: {:.4f}'.format(mAP))
    # print("recall: {:.3f}, precision: {:.3f}".format(rec_total.average, prec_total.average))
    # print("total_loss: {:.3f}, loss_xy: {:.3f}, loss_wh: {:.3f}, loss_conf: {:.3f}, loss_class: {:.3f}".format(
    #     val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average
    # ))
