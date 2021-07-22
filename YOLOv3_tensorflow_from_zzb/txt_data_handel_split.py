import os
import cv2
# import args
import random


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


def split_txt_data(dataset_txt, train_txt, val_txt):
    train_txt_data = open(train_txt, 'w')

    val_txt_data = open(val_txt, 'w')

    dataset_txt_data = []
    for data_line in open(dataset_txt, 'r'):
        dataset_txt_data.append(data_line)

    random.shuffle(dataset_txt_data)
    indx = 0
    for line in dataset_txt_data:
        if indx % 10 == 0:
            val_txt_data.write(line)
        else:
            train_txt_data.write(line)
        indx += 1
    train_txt_data.close()
    val_txt_data.close()


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def split_txt_data_for(dataset_txt, train_txt, val_txt, class_name_path):
    train_txt_data = open(train_txt, 'w')

    val_txt_data = open(val_txt, 'w')

    dataset_txt_data = []
    for data_line in open(dataset_txt, 'r'):
        dataset_txt_data.append(data_line)
    # 统计各个类别中图片的数量  限制是：每张图片里有且仅有一条数据
    classes = read_class_names(class_name_path)
    class_num = len(classes)
    for class_ in range(class_num):
        line_data_list = []
        for line in dataset_txt_data:
            line_ = line.strip('\n').split(' ')
            label = line_[1]
            if label == classes[class_]:
                line_data_list.append(line)
        print(classes[class_], ' num: ', len(line_data_list))
        # 多条数据
        if len(line_data_list) >= 2:
            # 随机打乱
            random.shuffle(line_data_list)
            index = 0
            for ln in line_data_list:
                if index % 9 == 0:
                    val_txt_data.write(ln)
                else:
                    train_txt_data.write(ln)
                index += 1
        elif len(line_data_list) == 1:
            # 就一条数据的情况下
            train_txt_data.write(line_data_list[0])
    train_txt_data.close()
    val_txt_data.close()


def spilt_txt_for_num_cow(dataset_txt, train_txt, val_txt, class_name_path):
    train_txt_data = open(train_txt, 'w')

    val_txt_data = open(val_txt, 'w')

    dataset_txt_data = []
    for data_line in open(dataset_txt, 'r'):
        dataset_txt_data.append(data_line)
    # 统计各个类别中图片的数量
    classes = read_class_names(class_name_path)
    print(classes)
    class_num = len(classes)

    for class_ in range(class_num):
        line_data_list = []
        num_class_count = 0
        for line in dataset_txt_data:
            line_ = line.strip('\n').split(' ')
            # print(line)
            bb_label_data = line_[1:]
            bboxs, labels = find_bbox_label(bb_label_data)
            for label in labels:
                if label == classes[class_]:
                    # line_data_list.append(line)
                    num_class_count += 1
        print(classes[class_], ' num: ', num_class_count)
        # 多条数据
        # if len(line_data_list) >= 2:
        #     # 随机打乱
        #     random.shuffle(line_data_list)
        #     index = 0
        #     for ln in line_data_list:
        #         if index % 9 == 0:
        #             val_txt_data.write(ln)
        #         else:
        #             train_txt_data.write(ln)
        #         index += 1
        # elif len(line_data_list) == 1:
        #     # 就一条数据的情况下
        #     train_txt_data.write(line_data_list[0])
    train_txt_data.close()
    val_txt_data.close()


def count_data_distribution(dataset_txt):
    dataset_txt_data = []
    for data_line in open(dataset_txt, 'r'):
        dataset_txt_data.append(data_line)
    # 统计各个类别中图片的数量
    classes = read_class_names(class_name_path)
    class_num = len(classes)
    for class_ in range(class_num):
        line_data_list = []
        for line in dataset_txt_data:
            line_ = line.strip('\n').split(' ')
            label = line_[1]
            if label == classes[class_]:
                line_data_list.append(line)
        print(classes[class_], ' num: ', len(line_data_list))


def check_pic_is_exist(data):
    pakua_data = open(data, 'r')
    line_datas = pakua_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        img_name = line_list[0]
        img = cv2.imread(img_name)

        if img is None:
            print(img_name)
    pakua_data.close()


def check_pic(txt):
    img_dir = '/home/zhouzhubin/NFS_AIDATA'
    # img_dir = '/home/zhouzhubin/data/cowrecognition/monitor_video/pickdir/0628/062801/'
    pakua_data = open(txt, 'r')
    line_datas = pakua_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(',')
        img_name = line_list[1]
        img_name = img_name[13:]  # 去除/mnt/aidata3/
        img_name = os.path.join(img_dir, img_name)
        print(img_name)
        img = cv2.imread(img_name)
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("test", img)
        cv2.waitKey(0)


def check_pic_(txt):
    img_dir = '/home/zhouzhubin/data/cowrecognition/monitor_video/pickdir/0628/062801/'
    pakua_data = open(txt, 'r')
    line_datas = pakua_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(',')
        img_name = line_list[0]
        img_name = os.path.join(img_dir, img_name)
        print(img_name)
        img = cv2.imread(img_name)
        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("test", img)
        cv2.waitKey(0)

# h 1422
def splice_pic(txt):
    save_dir = '/home/zhouzhubin/桌面/splice_pic'
    pakua_data = open(txt, 'r')
    line_datas = pakua_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        img_name1 = line_list[0]
        img_name2 = line_list[1]
        save_name = line_list[2]
        print(img_name1)
        img1 = cv2.imread(img_name1)
        w, h = img1.shape[1], img1.shape[0]
        img1 = img1[0:1422, 0:w, :]
        cv2.putText(img1, save_name[:-4], (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        img2 = cv2.imread(img_name2)
        w, h = img2.shape[1], img2.shape[0]
        img2 = img2[0:1422, 0:w, :]

        img = cv2.vconcat([img1, img2])

        # save splice pic
        cv2.imwrite(os.path.join(save_dir, save_name), img)

        cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("test", img)
        cv2.waitKey(0)


# 裁剪BCS数据
def BCS_cut_img(txt, save_dir):
    img_dir = '/home/zhouzhubin/sjht_data/'
    save_path = save_dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    txt_data = open(txt, 'r')
    line_datas = txt_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        img_name = line_list[0]
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        bb_label_data = line_list[1:]
        bboxs, labels = find_bbox_label(bb_label_data)
        print(img_name)
        name = img_name.split('/')[-1][:-4]
        num = 0
        count_num = 0
        for bbox in bboxs:
            count_num += 1
            # 建label文件夹
            label = labels[num]
            label_path = os.path.join(save_path, label)
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            img_cut = img[y1:y2, x1:x2, :]
            cut_img_name = name + '_%.4d.jpg' % count_num
            # cv2.imshow("test", img_cut)
            # cv2.waitKey(0)
            # save cut img
            cv2.imwrite(os.path.join(label_path, cut_img_name), img_cut)

            num += 1


def cut_img(txt, save_dir):
    img_dir = '/home/zhouzhubin/sjht_data/'

    model = txt[-5]
    save_path = os.path.join(save_dir, model)
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    txt_data = open(txt, 'r')
    line_datas = txt_data.readlines()
    for line_data in line_datas:
        line_list = line_data.strip('\n').split(' ')
        img_name = line_list[0]
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        bb_label_data = line_list[1:]
        bboxs, labels = find_bbox_label(bb_label_data)
        print(img_name)
        name = img_name.split('/')[-1][:-4]
        num = 0
        count_num = 0
        for bbox in bboxs:
            count_num += 1
            # 建label文件夹
            label = labels[num][4:]
            label_path = os.path.join(save_path, label)
            if not os.path.exists(label_path):
                os.mkdir(label_path)

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            img_cut = img[y1:y2, x1:x2, :]
            cut_img_name = name + '_' + model + '_%.4d.jpg' % count_num
            # cv2.imshow("test", img_cut)
            # cv2.waitKey(0)
            # save cut img
            cv2.imwrite(os.path.join(label_path, cut_img_name), img_cut)

            num += 1


if __name__ == '__main__':
    all_data = './data/my_data/bcs_20200709_all_data.txt'
    # all_data = './data/my_data/train_cow_20200211.txt'
    # 奶台号码牌+牛体标注数据
    txt_data = "./data/my_data/bcs_202007016_alldata_cow_num.txt"
    txt_data = "./data/my_data/bcs_202007018_testdata_cow_num.txt"
    txt_data = "./data/my_data/bcs_202007020_alldata_haikang.txt"
    txt_data = "./data/my_data/bcs_202007026_alldata_haikang.txt"
    naitai = './data/my_data/bcs_20200805_num.txt'
    dcf = './data/my_data/penlin_20200809_only_dcf.txt'
    qcj = './data/my_data/train_qcj_20200825_data.txt'
    # 查看数据标注
    check_label_data(qcj)
    # 将数据分为train集和test集
    train_data = './data/my_data/train_bcs_20200805_only_num.txt'
    val_data = './data/my_data/val_bcs_20200805_only_num.txt'
    # split_txt_data(naitai, train_data, val_data)

    # 根据具体的类别中的图片数目来分离出train集和val集，防止样本极不均衡所带来的训练问题
    # class_name_path = './data/my_data/class.txt'
    # classes = read_class_names(class_name_path)
    # print(classes)
    # split_txt_data_for(all_data, train_data, val_data, class_name_path)

    # 根据号码牌的类别来统计
    num_txt = './data/my_data/bcs_202007011_num_A.txt'
    class_name_path = 'data/my_data/cow_class.txt'
    # classes = read_class_names(class_name_path)
    num_train_data = './data/my_data/train_bcs_202007011_haikang.txt'
    num_val_data = './data/my_data/val_bcs_202007011_haikang.txt'
    # spilt_txt_for_num_cow(num_txt, num_train_data, num_val_data, class_name_path)
    # split_txt_data(num_txt, num_train_data, num_val_data)

    # 统计数据集中的分布情况
    data_all = './data/my_data/bcs_20200706_fixed_cap_data.txt'  # ./data/my_data/bcs_guding.txt  bcs_toudai.txt
    # count_data_distribution(data_all)

    # 查看数据0427和0531的视频里的
    # txt_data = '/home/zhouzhubin/NFS_AIDATA/bodyscore/shenfeng0427/cowlist.txt'
    # check_pic(txt_data)
    # txt_data = '/home/zhouzhubin/data/cowrecognition/monitor_video/pickdir/0628/062801/ckimg.list'
    # check_pic_(txt_data)
    # 拼接4月份和五月份有问题的图片
    splice_txt = '/home/zhouzhubin/桌面/cow_class.txt'
    # splice_pic(splice_txt)
    # 裁剪牛体数据
    # cut_txt = './data/my_data/train_bcs_20200709_all_data.txt'
    cut_txt = './data/my_data/val_bcs_20200709_all_data.txt'
    # BCS_cut_img_save_dir = '/home/zhouzhubin/NFS_AIDATA/bodyscore/BCS_CUT_IMG/train'
    BCS_cut_img_save_dir = '/home/zhouzhubin/NFS_AIDATA/bodyscore/BCS_CUT_IMG/val'
    # BCS_cut_img(cut_txt, BCS_cut_img_save_dir)


    # 查看奶台号码牌
    # naitai = './data/my_data/bcs_20200708_nt_data_A.txt'
    # naitai = './data/my_data/bcs_20200708_nt_data_B.txt'
    naitai = './data/my_data/bcs_202007011_num_A.txt'


    # check_label_data(naitai)

    # 根据标注数据裁剪A,B两种型号的奶台号码牌,用于作为分类的数据集
    data_save_dir = '/home/zhouzhubin/workspace/project/datasets/naitai'
    # cut_img(naitai, data_save_dir)

    # 分开带牛体和号码牌的数据, 并且将号码牌归为一类
    txt_data = "./data/my_data/bcs_202007016_alldata_cow_num.txt"
    train_data = './data/my_data/train_bcs_202007016_alldata_cow_num.txt'
    val_data = './data/my_data/val_bcs_202007016_alldata_cow_num.txt'
    # split_txt_data(txt_data, train_data, val_data)

    '''
    # 脏棉
    # 查看脏棉图片是否存在
    ZM_data = '/home/zhouzhubin/workspace/project/pytorch/Yet-Another-EfficientDet-Pytorch/txt_data/20200318_zm_train_data.txt'
    # check_pic_is_exist(ZM_data)

    train_data = '/home/zhouzhubin/workspace/project/pytorch/Yet-Another-EfficientDet-Pytorch/txt_data/train_zm_20200608_test.txt'
    val_data = '/home/zhouzhubin/workspace/project/pytorch/Yet-Another-EfficientDet-Pytorch/txt_data/val_zm_20200608_test.txt'
    split_txt_data(ZM_data, train_data, val_data)
    '''


