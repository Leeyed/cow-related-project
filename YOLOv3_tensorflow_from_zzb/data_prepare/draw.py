import cv2
import os

url = 'D:\PycharmProjects\lius\yolov3\dataset\ZVZ-real-512\Image'
with open('data.txt', 'r') as f:
    data = f.readlines()

img_names = os.listdir(url)
suffixs = ['.jpg', '.tiff', '.tif', '.jpeg']

for line in data:
    line = line.strip().split(',')
    assert len(line)%9==1
    bbox_num = len(line)//9
    candidates = [line[0]+suffix for suffix in suffixs]
    select = [candidate for candidate in candidates if candidate in img_names]
    assert len(select)==1
    img_url = os.path.join(url, select[0])
    print(img_url)
    img = cv2.imread(img_url)
    # height, width, channel
    print(img.shape)
    # draw rotated rectangle
    for i in range(bbox_num):
        pt1 = (min(int(line[1+9*i+0]),int(line[1+9*i+2]),int(line[1+9*i+4]),int(line[1+9*i+6])),
               min(int(line[1+9*i+1]),int(line[1+9*i+3]),int(line[1+9*i+5]),int(line[1+9*i+7])))
        pt2 = (max(int(line[1+9*i+0]),int(line[1+9*i+2]),int(line[1+9*i+4]),int(line[1+9*i+6])),
               max(int(line[1+9*i+1]),int(line[1+9*i+3]),int(line[1+9*i+5]),int(line[1+9*i+7])))
        cv2.rectangle(img, pt1, pt2, (0,0,255), 2)

        # pt1 = (int(line[1+9*i+0]), int(line[1+9*i+1]))
        # pt2 = (int(line[1+9*i+2]), int(line[1+9*i+3]))
        # pt3 = (int(line[1+9*i+4]), int(line[1+9*i+5]))
        # pt4 = (int(line[1+9*i+6]), int(line[1+9*i+7]))
        # print(pt1, pt2)
        # BGR
        # cv2.line(img, pt1, pt2, (0,0,255), 2)
        # cv2.line(img, pt2, pt3, (0,0,255), 2)
        # cv2.line(img, pt3, pt4, (0,0,255), 2)
        # cv2.line(img, pt4, pt1, (0,0,255), 2)

    cv2.imshow('img', img)
    cv2.waitKey()