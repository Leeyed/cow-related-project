import cv2
import os
import numpy as np


def readMaskData(img_dir:str):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    idx = np.where(img>128,0,255)
    ans = np.array(idx, dtype=np.uint8)
    return ans


def boxValid(mask:np.ndarray, center:tuple or list, dst_size:np.array):
    img_data = cv2.resize(mask, dst_size[::-1])
    if img_data[center[1]][center[0]]:
        return True
    else:
        return False