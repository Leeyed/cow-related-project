"""
tensorflow
"""
import tensorflow as tf
import torch

# deep sort
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from model import yolov3
import cv2
import time
import copy
from utils.nms_utils import gpu_nms
import numpy as np

from ls.networks import resnetface20
from ls.u2net import segmentModel
import pickle
from ls.utils_reg import prepare_image, letterbox_image
from torchvision import transforms as T
from PIL import Image
import os

from ls.tools import readMaskData, boxValid

# ft2 = cv2.freetype.createFreeType2()
# ft2.loadFontData('./ls/chinese.ttf', 0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# detection model
def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


class detectionModel:
    def __init__(self, model_path, num_class, input_size, anchors, threshold, iou_threshold):
        anchors = parse_anchors(anchors)
        self.input_size = input_size
        self.input_data = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.input_data, False)
        self.pred_boxes, self.pred_confs, self.pred_probs = yolo_model.predict(pred_feature_maps)

        self.pred_scores = self.pred_confs * self.pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(self.pred_boxes, self.pred_scores, num_class, max_boxes=150,
                                                       score_thresh=threshold,
                                                       nms_thresh=iou_threshold)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)

    def forward(self, img):
        height_ori, width_ori = img.shape[:2]
        img = cv2.resize(img, tuple(self.input_size))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        boxes_, scores_, labels_, pred_confs_, pred_probs_ = self.sess.run(
            [self.boxes, self.scores, self.labels, self.pred_confs, self.pred_probs],
            feed_dict={self.input_data: img})
        boxes_[:, 0] *= (width_ori / float(self.input_size[0]))
        boxes_[:, 2] *= (width_ori / float(self.input_size[0]))
        boxes_[:, 1] *= (height_ori / float(self.input_size[1]))
        boxes_[:, 3] *= (height_ori / float(self.input_size[1]))
        return boxes_, scores_, labels_, pred_confs_, pred_probs_


def prepare_image(img_path, dst_size, transform):
    try:
        img = Image.open(img_path)
    except IOError:
        raise Exception("Error: read %s fail" % img_path)
    new_img, _, _ = letterbox_image(img, dst_size)
    input_data = transform(new_img)
    return input_data


class registerModel:
    def __init__(self, config):
        self.register_model = resnetface20(input_shape=config['register_input_shape'])
        self.register_model.load_state_dict(torch.load(config['register_model_path'], map_location=config['device']))
        self.device = config['device']
        self.register_model.to(self.device)
        self.register_model.eval()
        self.input_shape = config['register_input_shape']
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean, std)])

    def forward(self, img):
        """
        cv2img -> Image
        """
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # input_data = prepare_image(image_path, (self.input_shape[2], self.input_shape[1]), self.transform)
        new_img, _, _ = letterbox_image(image, (self.input_shape[2], self.input_shape[1]))
        input_data = self.transform(new_img)
        input_data = input_data.unsqueeze(0).to(self.device)
        return self.register_model(input_data)[0]


@torch.no_grad()
def get_max_mean_median_similarity_from_ref(each_test, test_feature, ref_feature_matrix, ref_labels: list,
                                            ref_imgs: list):
    '''
    用于计算一张图片, 与对应嵌件的参考图片 以及所有的参考图片的进行对比, 求出相似度的[(最大值, path1, path2),(最小值, path1, path2),均值, 所有参考图片相似度均值最大的]
    :param each_test:  image url
    :param feature1:  当前图片的feature vector
    :param ref_feature_matrix:  所有ref中的图片vector    ref_feature_matrix 顺序 与 ref_imgs 顺序一致
    :param ref_labels:  所有ref的label
    :param ref_imgs:  所有 ref iamge url
    :return:
    '''
    all_data = []
    # test_feature 扩张成 test_feature_matrix
    test_feature_matrix = test_feature.expand_as(ref_feature_matrix)
    predicts = torch.cosine_similarity(test_feature_matrix, ref_feature_matrix).cpu().numpy().reshape(-1)

    max_id = np.argmax(predicts)
    all_data.append((each_test, ref_imgs[max_id], predicts[max_id]))

    each_label_num = len(predicts) // len(ref_labels)
    predicts = predicts.reshape((len(ref_labels), each_label_num))
    means = predicts.mean(axis=0)
    max_mean_id = np.argmax(means)
    all_data.append((means[max_mean_id], ref_labels[max_mean_id]))

    medians = np.median(predicts, axis=1)
    max_median_id = np.argmax(medians)
    all_data.append((medians[max_median_id], ref_labels[max_median_id]))

    return all_data


class trackModel:
    def __init__(self, config: dict):
        # segmentation model init
        self.segment_model = segmentModel(config)
        # registration init
        self.register_model = registerModel(config)
        # detection model init
        self.detect_model = detectionModel(config['detection_model_path'], config['num_class'],
                                           config['input_size'], config['anchors'],
                                           config['threshold'], config['iou_threshold'])
        # deep sort model
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

        self.encoder = gdet.create_box_encoder(config['track_model_path'], batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)
        # self.detections = []
        self.start_track = False
        self.track_id = None

        # load registered data
        # self.feature_dict={}
        f = open(config['registered_cow_dataset'], 'rb')
        feature_dict = pickle.load(f)
        f.close()
        for key, value in feature_dict.items():
            feature_dict[key] = torch.from_numpy(value).to(config['device'])
        # sorted_ref_feature.append(ref_feature_dict[each_pic].clone().detach().to(config['device']))
        self.ref_feature_keys = list(feature_dict.keys())
        self.ref_feature_tensor = torch.stack(list(feature_dict.values()))
        self.register_threshold = config['register_threshold']

    def trans_boxs(self, boxes):
        """
        trans [x0,y0,x1,y1] ==>> [x, y, w, h], x, y是左上角点
        :param boxes:
        :return:
        """
        boxs = []
        for box in boxes:
            x0, y0, x1, y1 = box
            s = (x1 - x0) * (y1 - y0)
            if s > 0:
                # x = (x0+x1)/2
                # y = (y0+y1)/2
                w = x1 - x0
                h = y1 - y0
                boxs.append([x0, y0, w, h])
        return boxs

    def calc_iou(self, pred_boxes, true_boxes):
        ''' Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
            shape_info: pred_boxes: [N, 4]
            true_boxes: [V, 4]
        '''
        # [N, 1, 4]
        pred_boxes = np.expand_dims(pred_boxes, -2)
        # [1, V, 4]
        true_boxes = np.expand_dims(true_boxes, 0)
        # [N, 1, 2] & [1, V, 2] ==> [N, V, 2]
        intersect_mins = np.maximum(pred_boxes[..., :2], true_boxes[..., :2])
        intersect_maxs = np.minimum(pred_boxes[..., 2:], true_boxes[..., 2:])
        intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
        # shape: [N, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 1, 2]
        pred_box_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
        # shape: [N, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # [1, V, 2]
        true_boxes_wh = true_boxes[..., 2:] - true_boxes[..., :2]
        # [1, V]
        true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]
        # shape: [N, V]
        iou = intersect_area / (pred_box_area + true_boxes_area - intersect_area + 1e-10)
        return iou

    @torch.no_grad()
    def forward(self, img):
        img_ori = copy.deepcopy(img)
        img_paint = copy.deepcopy(img)
        height_ori, width_ori = img_ori.shape[:2]

        # mask_data = readMaskData('./ls/mask_video.jpg')

        # detection
        boxes_, scores_, labels_, pred_confs_, pred_probs_ = self.detect_model.forward(img)

        # reg
        trackBoxes = []
        features = []
        cow_ids = []
        reg_confs = []
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width_ori, x1), min(height_ori, y1)
            s = (x1 - x0) * (y1 - y0)
            if s <= 0: continue
            # center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
            # if not boxValid(mask_data, center, img_ori.shape[:2]): continue
            img_cut = img_ori[int(y0):int(y1), int(x0):int(x1), :]
            masked_img = self.segment_model.forward(img_cut)
            test_feature = self.register_model.forward(masked_img)
            features.append(test_feature.cpu().numpy())

            test_feature_matrix = test_feature.expand_as(self.ref_feature_tensor)
            # if torch.cuda.is_available():
            predicts = torch.cosine_similarity(test_feature_matrix, self.ref_feature_tensor).view(-1)
            # else:
            #     predicts = torch.cosine_similarity(test_feature_matrix, self.ref_feature_tensor).reshape(-1)
            _id = torch.argmax(predicts)
            # 阈值设定
            # cow_ids.append(predicts[_id].detach().numpy())
            matched_img_name = self.ref_feature_keys[_id.cpu().numpy()].split('/')
            reg_conf = np.round(predicts[_id].cpu().numpy(),2)

            reg_confs.append(reg_conf)

            cow_ids.append(str(matched_img_name[-2])+':'+str(reg_conf))

            trackBoxes.append(boxes_[i])
            cv2.rectangle(img_paint, (x0, y0), (x1, y1), (0, 0, 255),4)

        # deep sort
        boxs = self.trans_boxs(trackBoxes)
        # features = self.encoder(img_ori, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, reg_conf, cow_id, 1.0, feature) for bbox, reg_conf, cow_id, feature in zip(boxs, reg_confs, cow_ids, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        if len(boxes)!= len(indices):
            print('before', len(boxes))
            print('after', len(indices))
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        track_boxes = []

        # track_flag = False
        for i, track in enumerate(self.tracker.tracks):
            print(i, track.is_confirmed(), track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            bbox = track.to_tlbr()
            track_boxes.append(bbox.tolist())

            # 显示中文
            from PIL import Image, ImageDraw, ImageFont
            img_PIL = Image.fromarray(cv2.cvtColor(img_paint, cv2.COLOR_BGR2RGB))
            font = ImageFont.truetype('./ls/chinese.ttf', 40)
            fillColor = (255, 255, 255)
            position = (int(bbox[0]), int(bbox[1]))
            draw = ImageDraw.Draw(img_PIL)
            draw.text(position, str(track.cow_id), font=font, fill=fillColor)
            # draw.text(position, str(track.track_id), font=font, fill=fillColor)
            img_paint = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

            cv2.rectangle(img_paint, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 4)
            # cv2.putText(img_paint, str(track.cow_id), (int(bbox[0]), int(bbox[1])), 0, 1, (0, 255, 0), 4)
            # if track.track_id != self.track_id: continue
        return img_paint
