import cv2
import os


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


def check_label_data(txt):
    img_dir = '/home/zhouzhubin/sjht_data/'
    pakua_data = open(txt, 'r')
    line_datas = pakua_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        img_name = line_list[0]
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        bb_label_data = line_list[1:]
        bboxs, labels = find_bbox_label(bb_label_data)
        print(img_name)
        num = 0
        for bbox in bboxs:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [255, 255, 0], 5)
            if labels[num].startswith('b'):
                cv2.putText(img, str(labels[num]), (int(bbox[0]), int(bbox[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            elif labels[num].startswith('n'):
                cv2.putText(img, str(labels[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            elif labels[num].startswith('c'):
                cv2.putText(img, str(labels[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            num += 1

        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        # img_name = os.path.join('./0903', img_name)
        # cv2.imwrite(img_name, img)
        cv2.imshow("test", img)
        cv2.waitKey(0)
    pakua_data.close()
    cv2.destroyAllWindows()


import numpy as np
def calc_iou_(pred_boxes, true_boxes):
    '''
    Maintain an efficient way to calculate the ios matrix using the numpy broadcast tricks.
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


def py_cpu_nms(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    # scores = dets[:, 4]
    # scores = dets[:, 4]
    # print('areas  ', areas)
    # print('scores ', scores)
    #
    # # 这边的keep用于存放，NMS后剩余的方框
    keep = []
    #
    # # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    # index = scores.argsort()[::-1]
    # 就只是计算当前一组内box的iou
    index = []
    [index.append(x) for x in range(len(dets))]
    # print(index)
    index = np.array(index)
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。
    max_iou = 0
    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        # print(keep)
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # print(x11, y11, x22, y22)
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        # print('overlaps is', overlaps)

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is', ious)
        if len(ious) > 0:
            iou = np.max(ious)
            if iou > max_iou:
                max_iou = iou

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        print(idx)

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
        # print(index)
    return max_iou

label_list = ["cow_back", "cow_left_belly", "cow_right_belly"]
def calc_iou(txt):
    max_iou = []
    data = open(txt, 'r')
    line_datas = data.readlines()
    for line_data in line_datas:
        list0 = []
        list1 = []
        list2 = []
        line_list = line_data.strip('\n').split(' ')

        bb_label_data = line_list[1:]
        bboxs, labels = find_bbox_label(bb_label_data)
        for i in range(len(labels)):
            if labels[i] == label_list[0]:
                list0.append(bboxs[i])
            elif labels[i] == label_list[1]:
                list1.append(bboxs[i])
            elif labels[i] == label_list[2]:
                list2.append(bboxs[i])
        if len(list0) > 1:
            # calc_iou_(list0, list0)
            max_iou.append(py_cpu_nms(list0, 0.0))
        if len(list1) > 1:
            # calc_iou_(list0, list0)
            max_iou.append(py_cpu_nms(list1, 0.0))
        if len(list2) > 1:
            # calc_iou_(list0, list0)
            max_iou.append(py_cpu_nms(list2, 0.0))
        print(line_list[0])
        print("line max iou: ", max(max_iou))

    print("max IOU: ", max(max_iou))


def py_cpu_nms_for_cut(dets, labels, thresh):
    print("ori dets: ", dets)
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    dets_ = np.array(dets)
    x1 = dets_[:, 0]
    y1 = dets_[:, 1]
    x2 = dets_[:, 2]
    y2 = dets_[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    # scores = dets[:, 4]
    # scores = dets[:, 4]
    # print('areas  ', areas)
    # print('scores ', scores)
    #
    # # 这边的keep用于存放，NMS后剩余的方框
    keep = []
    # 两个矩形框之间的IOU超过指定阈值时, 记录当前框的id
    over_iou_id = []
    #
    # # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    # index = scores.argsort()[::-1]
    # 就只是计算当前一组内box的iou
    index = []
    [index.append(x) for x in range(len(dets_))]
    # print(index)
    index = np.array(index)
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。
    max_iou = 0
    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        # print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # every time the first is the biggst, and add it directly

        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        # print(keep)
        # print('x1', x1[i])
        # print(x1[index[1:]])

        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # print(x11, y11, x22, y22)
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        # print('overlaps is', overlaps)

        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # print('ious is', ious)
        for k in range(len(ious)):
            if ious[k] >= thresh:
                over_iou_id.append(index[0])
                over_iou_id.append(index[k+1])

        if len(ious) > 0:
            iou = np.max(ious)
            if iou > max_iou:
                max_iou = iou

        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        # print(idx)

        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx + 1]  # because index start from 1
        # print(index)
    # 去除相同
    print(over_iou_id)
    over_iou_id = list(set(over_iou_id))
    print(over_iou_id)
    # 删除原list中,对应的下标的元素,然后生成新的list
    dets = [dets[i] for i in range(len(dets)) if(i not in over_iou_id)]
    if len(dets) != len(labels):
        print("去除后的: ", dets)
    # 删除原list中,对应的下标的元素,然后生成新的list
    labels = [labels[i] for i in range(len(labels)) if(i not in over_iou_id)]


    return dets, labels


# 根据IOU直接从标注平台截出牛的图片并保存
def cut_cow_pic(txt_data, save_dir, IOU=0.45):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = open(txt_data, 'r')
    line_datas = data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')

        bb_label_data = line_list[1:]
        bboxs, labels = find_bbox_label(bb_label_data)
        cut_bboxs, cut_labels = py_cpu_nms_for_cut(bboxs, labels, IOU)

        img_dir = '/home/zhouzhubin/sjht_data/'
        img_name = line_list[0]
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        num = 0
        for bbox in cut_bboxs:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), [255, 255, 0], 5)
            # if cut_labels[num].startswith('b'):
            #     cv2.putText(img, str(cut_labels[num]), (int(bbox[0]), int(bbox[1]) + 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            # elif cut_labels[num].startswith('n'):
            #     cv2.putText(img, str(cut_labels[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            # elif cut_labels[num].startswith('c'):
            #     cv2.putText(img, str(cut_labels[num][4:]), (int(bbox[0]), int(bbox[1]) + 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)
            cut_img = img[y1:y2, x1:x2, :]
            label_index = label_list.index(cut_labels[num])
            path = os.path.join(save_dir, str(label_index))
            if not os.path.exists(path):
                os.mkdir(path)
            cut_img_name = "%s_%.4d.jpg" % (img_name.split('/')[-1][:-4], num)
            cv2.imwrite(os.path.join(path, cut_img_name), cut_img)
            num += 1

        # cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)


import random
def split_txt_data(dataset_txt, train_txt, val_txt):
    train_txt_data = open(train_txt, 'w')

    val_txt_data = open(val_txt, 'w')

    dataset_txt_data = []
    for data_line in open(dataset_txt, 'r'):
        dataset_txt_data.append(data_line)

    random.shuffle(dataset_txt_data)
    indx = 0
    id_t = 0
    id_v = 0
    for line in dataset_txt_data:
        if indx % 10 == 0:
            line = str(id_v) + ' ' + line
            val_txt_data.write(line)
            id_v += 1
        else:
            line = str(id_t) + ' ' + line
            train_txt_data.write(line)
            id_t += 1
        indx += 1
    train_txt_data.close()
    val_txt_data.close()


if __name__ == '__main__':
    txt_data = "./data/my_data/qcj_20200827_data.txt"
    check_label_data(txt_data)
    calc_iou(txt_data)

    # 截图
    iou = 0.1
    save_dir = '/home/zhouzhubin/NFS_AIDATA/test2/cow_qcj_data/20200828_1'
    # cut_cow_pic(txt_data, save_dir, iou)

    # 分数据+id
    train = "./data/my_data/train_qcj_20200827_data_with_id.txt"
    val = "./data/my_data/val_qcj_20200827_data_with_id.txt"
    # split_txt_data(txt_data, train, val)


