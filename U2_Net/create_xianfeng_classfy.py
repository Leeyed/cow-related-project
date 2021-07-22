import cv2
import os
import random
import numpy as np
nfsDir = '/home/zhouzhubin/sjht_data/'


def create_data(txt, saveDir, train_val_scale=0.8, size=128, selectNum=5):
    txt_data = open(txt, 'r')
    line_datas = txt_data.readlines()
    num = 0
    size_list = []
    train_val_num = int(train_val_scale*len(line_datas))
    for i, line_data in enumerate(line_datas):
        if i <= train_val_num:
            save_dir = os.path.join(saveDir, 'train')
        else:
            save_dir = os.path.join(saveDir, 'val')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        line_list = line_data.strip('\n').split('#')
        print(line_list)
        imgPath = os.path.join(nfsDir, line_list[0])
        imgname = line_list[0].split('/')[-1][:-4]
        img = cv2.imread(imgPath)
        ori_w, ori_h = img.shape[1], img.shape[0]
        line_list = line_list[1:]
        for each in line_list:
            each_list = each.split(' ')
            label = each_list[0]

            label_path = os.path.join(save_dir, label)
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            points_list = each_list[1:]
            points = []
            for i in range(int(len(points_list) / 2)):
                x = int(points_list[2 * i])
                y = int(points_list[2 * i + 1])
                points.append([x, y])
            for num in range(selectNum):
                random.shuffle(points)
                x_c = int((points[0][0] + points[1][0])/2)
                y_c = int((points[0][1] + points[1][1])/2)

                x0 = max(0, int(x_c - size/2))
                y0 = max(0, int(y_c - size/2))
                x1 = min(ori_w, int(x_c + size / 2))
                y1 = min(ori_h, int(y_c + size / 2))
                cut_img = img[y0: y1, x0: x1, :]
                cut_img_name = "%s_%s_%.2d.jpg" % (imgname, label, num)
                cv2.imwrite(os.path.join(label_path, cut_img_name), cut_img)
                print(cut_img_name)

txt = 'xianfeng20210323.txt'

saveDir = '/home/zhouzhubin/workspace/project/datasets/xianfeng/20210323'
create_data(txt, saveDir)


# 工件分割
def get_image(path):  # 获取图片
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def Gaussian_Blur(gray):  # 高斯去噪(去除图像中的噪点)
    """
    高斯模糊本质上是低通滤波器:
    输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和

    高斯矩阵的尺寸和标准差:
    (9, 9)表示高斯矩阵的长与宽，标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。
    高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大。
    """


    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def Sobel_gradient(blurred):
    """
     索比尔算子来计算x、y方向梯度
     关于算子请查看:https://blog.csdn.net/wsp_1138886114/article/details/81368890
    """


    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradX, gradY, gradient


def Thresh_and_blur(gradient):  # 设定阈值
    blurred = cv2.GaussianBlur(gradient, (5, 5), 0)
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # (_, thresh) = cv2.threshold(blurred, 0, 200, cv2.THRESH_BINARY)
    """
    cv2.threshold(src,thresh,maxval,type[,dst])->retval,dst (二元值的灰度图)
    src：  一般输入灰度图
	thresh:阈值，
	maxval:在二元阈值THRESH_BINARY和
	       逆二元阈值THRESH_BINARY_INV中使用的最大值 
	type:  使用的阈值类型
    返回值  retval其实就是阈值 
	"""
    return thresh


def Thresh_and_blur_(gradient):  # 设定阈值
    blurred = cv2.GaussianBlur(gradient, (5, 5), 0)
    (_, thresh) = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(blurred,40,80)
    cv2.imshow('ca',canny)

    # (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    """
    cv2.threshold(src,thresh,maxval,type[,dst])->retval,dst (二元值的灰度图)
    src：  一般输入灰度图
	thresh:阈值，
	maxval:在二元阈值THRESH_BINARY和
	       逆二元阈值THRESH_BINARY_INV中使用的最大值 
	type:  使用的阈值类型
    返回值  retval其实就是阈值 
	"""
    return canny


def image_morphology(thresh):
    """
     建立一个椭圆核函数
     执行图像形态学, 细节直接查文档，很简单
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('rr',closed)

    # closed = cv2.erode(closed, (5,5))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    closed = cv2.morphologyEx(closed, cv2.MORPH_DILATE, kernel1)
    # closed1 = cv2.dilate(closed, (120,120))
    # cv2.imshow('rr1', closed1)
    # cv2.waitKey(0)
    # closed = cv2.erode(closed, None, iterations=4)
    # closed = cv2.dilate(closed, None, iterations=4)
    return closed


def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    (cnts, _) = cv2.findContours(closed.copy(),
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    zc = cv2.arcLength(c, True)
    # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


def drawcnts_and_cut(original_img, box):  # 目标图像裁剪
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    return draw_img, crop_img


def walk():
    img_path = r'/home/zhouzhubin/sjht_data/images/sjht/ai-product-injection-mold-inserts/office-33/pic/examine/20210318161914-568/camera2_2021-03-18_08_26_13_356966.jpg'
    img_path = r'/home/zhouzhubin/sjht_data/images/sjht/ai-product-injection-mold-inserts/office-33/pic/examine/20210318161914-568/camera1_2021-03-18_08_19_50_365534.jpg'
    img_path = r'/home/zhouzhubin/workspace/project/datasets/u2net/xianfeng_20210323/train/ori_pic/camera1_2021-03-18_08_28_28_022852.jpg.jpg'
    save_path = r'./cat_save.png'
    original_img, gray = get_image(img_path)
    blurred = Gaussian_Blur(gray)
    # gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(blurred)
    closed = image_morphology(thresh)


    mask_gray = (blurred*np.where(closed == 0, 1, 0)).astype(np.uint8)

    mask_gray_thresh = Thresh_and_blur_(mask_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_gray_thresh = cv2.morphologyEx(mask_gray_thresh, cv2.MORPH_CLOSE, kernel)

    # box = findcnts_and_box_point(closed)
    # draw_img, crop_img = drawcnts_and_cut(original_img, box)

    # 暴力一点，把它们都显示出来看看
    cv2.namedWindow("original_img", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("GaussianBlur", cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow("gradX", cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow("gradY", cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow("final", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("thresh", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("closed", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("mask_gray", cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow("mask_gray_thresh", cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow("draw_img", cv2.WND_PROP_FULLSCREEN)
    # cv2.namedWindow("crop_img", cv2.WND_PROP_FULLSCREEN)

    cv2.imshow('original_img', original_img)
    cv2.imshow('GaussianBlur', blurred)
    # cv2.imshow('gradX', gradX)
    # cv2.imshow('gradY', gradY)
    # cv2.imshow('final', gradient)
    cv2.imshow('thresh', thresh)
    cv2.imshow('closed', closed)
    cv2.imshow('mask_gray', mask_gray)
    cv2.imshow('mask_gray_thresh', mask_gray_thresh)
    # cv2.imshow('draw_img', draw_img)
    # cv2.imshow('crop_img', crop_img)
    cv2.waitKey(0)
    # cv2.imwrite(save_path, crop_img)


# if __name__ == '__main__':
#     walk()
