import os
import torch
import cv2
import numpy as np
import pickle
import time
import math
import copy
from PIL import Image

from torchvision import transforms as T


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    shift = [(w - nw) // 2, (h - nh) // 2]
    return new_image, scale, shift


def prepare_image(img_path, dst_size, transform):
    try:
        img = Image.open(img_path)
    except IOError:
        raise Exception("Error: read %s fail" % img_path)
    new_img, _, _ = letterbox_image(img, dst_size)
    input_data = transform(new_img)
    return input_data


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def get_dirs_all_data(imgdir):
    """
    获取imgdir下所有label内的图片, 返回所有图片的路径
    """
    img_list = []
    for each in os.listdir(imgdir):
        # 去除backgrond, wxqy
        if each == 'background' or each == 'wxqy' or each == 'hscfxpmgp' or each == 'hsbq':
            continue

        sub_dir = os.path.join(imgdir, each)
        files = os.listdir(sub_dir)
        paths = []
        for each_file in files:
            # 去除 'Thumbs.db' 文件
            if 'Thumbs.db' in each_file: continue
            paths.append(os.path.join(sub_dir, each_file))
        img_list.extend(paths)
    return img_list


def cnt_area(contour):
    xmin = np.min(contour[:, :, 0])
    ymin = np.min(contour[:, :, 1])
    xmax = np.max(contour[:, :, 0])
    ymax = np.max(contour[:, :, 1])
    return np.sqrt(np.square(xmax - xmin) + np.square(ymax - ymin))


def save_output_mask_rect(image_name, predict, save_dir):
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = np.where(predict_np < 0.5, 0, 1).astype(np.uint8)

    # opencv2.x 和 opencv3.x api 有区别
    try:
        contours, _ = cv2.findContours(predict_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv2.findContours(predict_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cnt_area, reverse=True)
    bkg = np.zeros(predict_np.shape).astype(np.uint8)
    cv2.drawContours(bkg, contours, 0, (1, 1, 1), -1)

    cv_img = cv2.imread(image_name)
    mask_np = cv2.resize(bkg, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    try:
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        __, contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cnt_area, reverse=True)
    rotated_box = cv2.minAreaRect(contours[0])

    mask_np_125 = np.where(mask_np == 0, 125, 0).astype(np.uint8)
    mask_np_1 = mask_np[:, :, np.newaxis]
    mask_np_1_125 = mask_np_125[:, :, np.newaxis]
    mask_np_3 = np.concatenate((mask_np_1, mask_np_1, mask_np_1), axis=-1)
    mask_np_3_125 = np.concatenate((mask_np_1_125, mask_np_1_125, mask_np_1_125), axis=-1)
    pcd_img = mask_np_3 * cv_img + mask_np_3_125

    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # 以mask旋转框的中心点，为新图片的中心点进行填充
    new_w, new_h = int(size[0]), int(size[1])
    c_x, c_y = int(center[0]), int(center[1])

    length = int(math.sqrt((new_h) ** 2 + (new_w) ** 2))
    new_img_ = copy.deepcopy(pcd_img)
    center_x = c_x

    ori_w, ori_h = pcd_img.shape[1], pcd_img.shape[0]
    # left
    if c_x < length / 2:
        tmp = int(length / 2) - c_x
        new_img_ = cv2.copyMakeBorder(new_img_, 0, 0, tmp, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])
        center_x = int(length / 2)
    # right
    if length / 2 > (ori_w - c_x):
        tmp = int(length / 2 - (ori_w - c_x))
        new_img_ = cv2.copyMakeBorder(new_img_, 0, 0, 0, tmp, cv2.BORDER_CONSTANT, value=[125, 125, 125])

    # up
    center_y = c_y
    if c_y < length / 2:
        tmp = int(length / 2) - c_y
        new_img_ = cv2.copyMakeBorder(new_img_, tmp, 0, 0, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])
        center_y = int(length / 2)

    # down
    if length / 2 > (ori_h - c_y):
        tmp = int(length / 2 - (ori_h - c_y))
        new_img_ = cv2.copyMakeBorder(new_img_, 0, tmp, 0, 0, cv2.BORDER_CONSTANT, value=[125, 125, 125])

    height, width = new_img_.shape[0], new_img_.shape[1]
    if angle < -45:
        angle = angle + 90
        size = (size[1], size[0])
    center = (center_x, center_y)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    img_rot = cv2.warpAffine(new_img_, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    # 根据宽高看是否旋转90度
    img_crop_w, img_crop_h = img_crop.shape[1], img_crop.shape[0]
    if img_crop_w > img_crop_h:
        img_crop = np.rot90(img_crop)
    # image process complete

    img_name = image_name.split(os.sep)[-1]
    # class_label = image_name.split(os.sep)[-2]
    #
    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1, len(bbb)):
    #     imidx = imidx + "." + bbb[i]
    #
    # class_label_path = os.path.join(save_dir, class_label) + os.sep
    # if not os.path.exists(class_label_path):
    #     os.makedirs(class_label_path)

    cv2.imwrite(os.path.join(save_dir, img_name), img_crop)
    # cv2.imwrite(save_dir + imidx + '.jpg', img_crop)


@torch.no_grad()
def save_infer_feature(img_dir, pkl_path, model, config):
    input_shape = config['input_shape']
    img_list = []

    floders = os.listdir(img_dir)
    print('img_dir', img_dir)
    # floders.remove('feature.pkl')
    for folder in floders:
        if folder.endswith('.pkl'): continue
        imgs = os.listdir(os.path.join(img_dir, folder))
        imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
        for img in imgs:
            if not img.endswith('.jpg'): continue
            if '副本' in img: continue
            img_list.append(os.path.join(img_dir, folder, img))
    print('len(img_list)', len(img_list))
    print(img_list[:2])
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean, std)])
    batch_size = 256
    save_feature_data = {}
    cnt = 0
    batch_data = []
    batch_keys = []

    while (cnt < len(img_list)):
        image_path = img_list[cnt]
        input_data = prepare_image(image_path, (input_shape[2], input_shape[1]), transform)
        input_data = input_data.unsqueeze(0).to(config['device'])
        batch_keys.append(image_path)
        batch_data.append(input_data)

        cnt += 1
        # python [int] infinity
        if cnt%batch_size == 0 or cnt == len(img_list):
            t1 = time.time()
            batch_data = torch.cat(batch_data)
            features = model.forward(batch_data)
            print('batch infer time: %f' % (time.time() - t1))

            for j, feature_key in enumerate(batch_keys):
                save_feature_data[feature_key] = features[j].cpu().numpy()

            batch_data = []
            batch_keys = []

    # 将输出的保存为pkl
    print(pkl_path)
    save_data_output = open(pkl_path, 'wb')
    pickle.dump(save_feature_data, save_data_output)
    save_data_output.close()


def readMaskData(dir:str):
    imgs = os.listdir(dir)
    imgs = list(filter(lambda x:x.endswith('.jpg'), imgs))
    ans = {}
    for img_name in imgs:
        img = cv2.imread(os.path.join(dir, img_name), cv2.IMREAD_GRAYSCALE)
        idx = np.where(img>128,0,255)
        img_name = img_name[:4]
        ans[img_name] = np.array(idx, dtype=np.uint8)
    return ans


def boxValid(data:dict, center:tuple or list, video_name:str, dst_size:np.array):
    if video_name[:4] not in data.keys():
        return True
    before_img = data.get(video_name[:4])
    img_data = cv2.resize(before_img, dst_size[::-1])
    if img_data[center[1]][center[0]]:
        return True
    else:
        return False


if __name__ == '__main__':
    aaa = readMaskData(r'D:\PycharmProjects\lius\0_projects\videoCut_cowRec_bodaiRec_cowSeg_Regisiter\mask_image_data')
    pass
