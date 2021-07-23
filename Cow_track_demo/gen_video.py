import os
import cv2
import numpy as np

path = '../res'
# filelist = os.listdir(path)
# filelist.sort()

fps = 24 #视频每秒24帧
size = (960, 540) #需要转为视频的图片的尺寸
#可以使用cv2.resize()进行修改

video = cv2.VideoWriter("VideoTest1.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#视频保存在当前目录下

for index in range(5594):
# for index in range(100):
    img = os.path.join(path, str(index+1)+'.jpg')
    print(img)
    img = cv2.imread(img)
    video.write(img)

video.release()
# cv2.destroyAllWindows()