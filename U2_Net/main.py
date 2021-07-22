import cv2 as cv
import numpy as np
import os


def show_img(img, name='', zoom=0.5):
    H, W = img.shape[:2]
    imgresize = cv.resize(img, (int(zoom * W), int(zoom * H)), cv.INTER_LINEAR)
    cv.imshow(name, imgresize)


def get_max_Contour(Contours):
    arealist = []
    for Contour in Contours:
        arealist.append(cv.contourArea(Contour))
    return arealist.index(max(arealist))


def get_linesregion(img, mask):
    img = cv.GaussianBlur(img, (5, 5), 0)
    canny = cv.Canny(img, 40, 80)
    canny = cv.multiply(canny, mask)
    element1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
    linesregion = cv.morphologyEx(canny, cv.MORPH_CLOSE, element1)
    return linesregion


def get_points(linesregion):
    # def cnt_area(contour):
    #     area = cv.contourArea(contour)
    #     return area

    def cnt_area(contour):
        xmin = np.min(contour[:, :, 0])
        ymin = np.min(contour[:, :, 1])
        xmax = np.max(contour[:, :, 0])
        ymax = np.max(contour[:, :, 1])
        return np.sqrt(np.square(xmax - xmin) + np.square(ymax - ymin))

    points = []
    try:
        contours, _ = cv.findContours(linesregion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv.findContours(linesregion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cnt_area, reverse=True)
    for i in range(3):
        try:
            M = cv.moments(contours[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX, cY])
        except:
            break
    return points


def get_boxes(points, boxsize):
    boxes = []
    for point in points:
        boxes.append([int(point[0] - (boxsize / 2)), int(point[1] - (boxsize / 2)), int(point[0] + (boxsize / 2)),
                      int(point[1] + (boxsize / 2))])
    return boxes


def get_line_points(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _, bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = np.zeros(img.shape, dtype=np.uint8)
    bin = 255 - bin
    try:
        contours, _ = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(mask, contours, get_max_Contour(contours), (1, 1, 1), -1)
    element1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
    mask = cv.morphologyEx(mask, cv.MORPH_ERODE, element1)
    linesregion = get_linesregion(img, mask)
    points = get_points(linesregion)
    return points


import copy
if __name__ == '__main__':
    imglist = os.listdir('/home/zhouzhubin/workspace/project/datasets/u2net/xianfeng_20210323/val/ori_pic')

    for name in imglist:
        path = os.path.join('/home/zhouzhubin/workspace/project/datasets/u2net/xianfeng_20210323/val/ori_pic', name)

        # img = cv.imread(path, 0)
        # _, bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # mask = np.zeros(img.shape, dtype=np.uint8)
        # bin = 255 - bin
        # try:
        #     contours, _ = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # except:
        #     __, contours, _ = cv.findContours(bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #
        # cv.drawContours(mask, contours, get_max_Contour(contours), (1, 1, 1), -1)
        # element1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (41, 41))
        # mask = cv.morphologyEx(mask, cv.MORPH_ERODE, element1)
        # linesregion = get_linesregion(img, mask)
        # points = get_points(linesregion)

        img = cv.imread(path)
        points = get_line_points(img)

        boxes = get_boxes(points, 128)
        img = cv.imread(path)
        img_ori = copy.deepcopy(img)
        for i, box in enumerate(boxes):
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
            cv.circle(img, (points[i][0], points[i][1]), 4, (255, 0, 0), -1)
        cv.imshow("img_ori", img_ori)
        show_img(img, 'img', 1)
        cv.waitKey(0)

