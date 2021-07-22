import os
import sys
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from PIL import Image
import cv2
import random
from modules_reg_plus.utils_reg import letterbox_image
# import albumentations as A

# sys.path 返回的是一个列表！
# 自己写的脚本不在同一个目录下，开头加sys.path.append('xxx')：
sys.path.append("..")
torch.multiprocessing.set_sharing_strategy('file_system')

class DatasetTriplelet(data.Dataset):
    '''
    batch hard strategy:
    randomly sampling P classes, randomly sampling K images of each class, resulting in a batch of PK images
    select the hardest positive and the hardest negative samples within the batch
    P = 18 K = 4
    '''

    def __init__(self, root, phase='train', input_shape=(3, 128, 128), P=18, K=4):
        self.phase = phase
        self.input_shape = input_shape
        self.sample_nums = 1000
        self.root = root
        self.dir_list = os.listdir(root)
        self.P = P
        self.K = K

        label_dict = {}
        for i, each in enumerate(self.dir_list):
            label_dict[each] = i
        self.label_dict = label_dict

        imgs_list_dict = {}
        for i, each in enumerate(self.dir_list):
            img_list = os.listdir(os.path.join(self.root, each))
            if 'Thumbs.db' in img_list: img_list.remove('Thumbs.db')
            if 'Thumbs.db:encryptable' in img_list: img_list.remove('Thumbs.db:encryptable')
            imgs_list_dict[each] = img_list
        self.imgs_list_dict = imgs_list_dict

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=180),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    # 随机裁剪,只支持opencv读出来的图片
    def cut_random(self, imgrgb, cut_rate=0.02):
        h, w, c = imgrgb.shape

        cut_h_u = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_h_d = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_w_l = random.choice(range(1, max(int(w * cut_rate), 2)))
        cut_w_r = random.choice(range(1, max(int(w * cut_rate), 2)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]
        return imgrgb

    # 模糊处理
    def img_blur(self, imgrgb):
        choice_list = [3, 5]

        my_choice = random.sample(choice_list, 1)
        img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
        return img_blur

    def __getitem__(self, index):

        choosen_clses = random.sample(self.dir_list, self.P)

        data_list = []
        label_list = []
        img_list = []
        for each in choosen_clses:
            label = self.label_dict[each]
            choosen_files = random.sample(self.imgs_list_dict[each], self.K)

            for each_file in choosen_files:
                img_path = os.path.join(self.root, each, each_file)
                img_list.append(img_path)
                img = None
                try:
                    if self.phase == 'train':
                        img = cv2.imread(img_path)
                    else:
                        img = Image.open(img_path)

                except IOError:
                    print(f"Error: read {img_path} fail")

                # gray_img = img.convert('LA')
                # gray_img = PIL.ImageOps.equalize(gray_img, mask=None)
                # img = PIL.ImageOps.equalize(img, mask=None)
                # 添加随机裁剪
                if self.phase == 'train':
                    # 随机模糊
                    if random.random() > 1:
                        img = self.img_blur(img)
                    # img = self.cut_random(img)
                    # 将opencv读出来的图像转化为BGR
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

                img, _, _ = letterbox_image(img, self.input_shape[1:])
                img_tensor = self.transforms(img)
                data_list.append(img_tensor)
                label_list.append(label)

        # new_label = max(label_list) + 1
        #
        # cnt = 0
        # pairs = []
        # P_idx = list(range(self.P))
        # # while(cnt < self.P*(self.P-1)):
        # while(cnt < 60):
        #     pair = random.sample(P_idx, 2)
        #     if pair not in pairs:
        #         pairs.append(pair)
        #         cnt += 1
        #
        # for i, pair in enumerate(pairs):
        #     cls1_img = data_list[pair[0]*self.K:(pair[0]+1)*self.K]
        #     cls2_img = data_list[pair[1]*self.K:(pair[1]+1)*self.K]
        #     for j in range(self.K):
        #         new_img = (cls1_img[j] + cls2_img[j])/2.0
        #         data_list.append(new_img)
        #         label_list.append(new_label + i)

        # tensor_list = [self.transforms(img) for img in data_list]
        # print(tensor_list)
        # img_tensor = self.transforms(img)

        # for i in range(self.P - 1):
        #     cls1_img_tensor = data_list[i*self.K:(i+1)*self.K]
        #     cls2_img_tensor = data_list[(i+1)*self.K:(i+2)*self.K]
        #     for j in range(self.K):
        #         new_img_tensor = (cls1_img_tensor[j] + cls2_img_tensor[j])/2.0
        #         data_list.append(new_img_tensor)
        #         label_list.append(new_label + i)
        return data_list, label_list, img_list

    # def __getitem__(self, index):
    #     p_dir = self.dir_list[index]
    #     n_list = copy.deepcopy(self.dir_list)
    #     n_list.pop(index)
    #     n_dir = random.choice(n_list)

    #     p_path = os.path.join(self.root, p_dir)
    #     n_path = os.path.join(self.root, n_dir)

    #     a = random.sample(os.listdir(p_path), 2)[0]
    #     p = random.sample(os.listdir(p_path), 2)[1]
    #     n = random.choice(os.listdir(n_path))

    #     try:
    #         img_path = os.path.join(p_path, a)
    #         a_img = Image.open(img_path)
    #     except IOError:
    #         print(f"Error: read {img_path} fail")

    #     try:
    #         img_path = os.path.join(p_path, p)
    #         p_img = Image.open(img_path)
    #     except IOError:
    #         print(f"Error: read {img_path} fail")

    #     try:
    #         img_path = os.path.join(n_path, n)
    #         n_img = Image.open(img_path)
    #     except IOError:
    #         print(f"Error: read {img_path} fail")

    #     a_img, _, _ = letterbox_image(a_img, self.input_shape[1:])
    #     p_img, _, _ = letterbox_image(p_img, self.input_shape[1:])
    #     n_img, _, _ = letterbox_image(n_img, self.input_shape[1:])

    #     data_a = self.transforms(a_img)
    #     data_p = self.transforms(p_img)
    #     data_n = self.transforms(n_img)

    #     data = torch.stack([data_a, data_p, data_n])
    #     return data.float()

    def __len__(self):
        return len(self.dir_list)
        # return self.P*self.K


# add by zhou
class DatasetTriplelet_new(data.Dataset):
    '''
    batch hard strategy:
    randomly sampling P classes, randomly sampling K images of each class, resulting in a batch of PK images
    select the hardest positive and the hardest negative samples within the batch
    P = 18 K = 4
    '''

    def __init__(self, root, phase='train', input_shape=(3, 128, 128), P=18, K=4):
        self.phase = phase
        self.input_shape = input_shape
        self.sample_nums = 1000
        self.root = root
        self.dir_list = os.listdir(root)
        self.P = P
        self.K = K

        label_dict = {}
        for i, each in enumerate(self.dir_list):
            label_dict[each] = i
        self.label_dict = label_dict

        all_imgs_list = []
        all_imgs_label_list = []
        for i, each in enumerate(self.dir_list):
            img_list = os.listdir(os.path.join(self.root, each))
            if 'Thumbs.db' in img_list: img_list.remove('Thumbs.db')
            if 'Thumbs.db:encryptable' in img_list: img_list.remove('Thumbs.db:encryptable')
            for each_img in img_list:
                all_imgs_list.append(os.path.join(each, each_img))
                all_imgs_label_list.append(each)

        self.all_imgs_list = all_imgs_list
        self.all_imgs_label_list = all_imgs_label_list

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=180),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    # 随机裁剪,只支持opencv读出来的图片
    def cut_random(self, imgrgb, cut_rate=0.02):
        h, w, c = imgrgb.shape

        cut_h_u = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_h_d = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_w_l = random.choice(range(1, max(int(w * cut_rate), 2)))
        cut_w_r = random.choice(range(1, max(int(w * cut_rate), 2)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]
        return imgrgb

    # 模糊处理
    def img_blur(self, imgrgb):
        choice_list = [3, 5]

        my_choice = random.sample(choice_list, 1)
        img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
        return img_blur

    def __getitem__(self, index):

        img_path = os.path.join(self.root, self.all_imgs_list[index])
        label = self.label_dict[self.all_imgs_label_list[index]]
        img = None
        try:
            if self.phase == 'train':
                img = cv2.imread(img_path)
            else:
                img = Image.open(img_path)

        except IOError:
            print(f"Error: read {img_path} fail")

        # gray_img = img.convert('LA')
        # gray_img = PIL.ImageOps.equalize(gray_img, mask=None)
        # img = PIL.ImageOps.equalize(img, mask=None)
        # 添加随机裁剪
        if self.phase == 'train':
            # 随机模糊
            if random.random() > 1:
                img = self.img_blur(img)
            img = self.cut_random(img)
            # 将opencv读出来的图像转化为BGR
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

        img, _, _ = letterbox_image(img, self.input_shape[1:])
        img_tensor = self.transforms(img)
        return img_tensor.float(), label

    def __len__(self):
        return len(self.all_imgs_list)


# add by zhou
class DatasetCircleLoss(data.Dataset):
    '''
    batch hard strategy:
    randomly sampling P classes, randomly sampling K images of each class, resulting in a batch of PK images
    select the hardest positive and the hardest negative samples within the batch
    P = 18 K = 4
    '''

    def __init__(self, root, phase='train', input_shape=(3, 128, 128), P=18, K=4):
        self.phase = phase
        self.input_shape = input_shape
        self.sample_nums = 1000
        self.root = root
        self.dir_list = os.listdir(root)
        self.P = P
        self.K = K

        label_dict = {}
        for i, each in enumerate(self.dir_list):
            label_dict[each] = i
        self.label_dict = label_dict

        all_batch_img_list = []
        # 随机选择P个类别嵌件
        all_label_list = os.listdir(root)
        random.shuffle(all_label_list)
        # self.P组
        num = len(all_label_list) // self.P
        for i in range(num):
            select_label = all_label_list[self.P * i:self.P * (i + 1)]
            # P个类别的嵌件图片
            P_label_pics_list = []
            # P个类别中最小个数的嵌件个数
            label_num_min = 0
            for insert in select_label:
                # if '+' in insert: continue
                insert_path = os.path.join(root, insert)
                files = os.listdir(insert_path)
                if len(files) > self.sample_nums:
                    selects = random.sample(files, self.sample_nums)
                else:
                    selects = files
                # 随机打乱
                random.shuffle(selects)
                # 删除带有Thumbs.db的文件
                new_selects = []  # 带图片全路径的
                for each_pic in selects:
                    if 'Thumbs.db' in each_pic: continue
                    new_selects.append(os.path.join(insert_path, each_pic))
                # 将每个嵌件类别分为 以K张图片的list集
                pic_num = len(new_selects) // self.K
                single_label_pic = []
                for j in range(pic_num):
                    single_label_pic.append(new_selects[self.K * j:self.K * (j + 1)])
                P_label_pics_list.append(single_label_pic)
                # 判断最小个数的嵌件
                if label_num_min == 0:
                    label_num_min = pic_num

                if label_num_min > pic_num:
                    label_num_min = pic_num

            # 将数据整合成[[P*K个元素],[P*K个元素], ...]
            batch_img = []
            for k in range(label_num_min):
                one_img = []
                for l in range(self.P):
                    one_img.extend(P_label_pics_list[l][k])

                batch_img.append(one_img)

            all_batch_img_list.extend(batch_img)

        self.all_batch_img_list = all_batch_img_list

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=180),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    # 随机裁剪,只支持opencv读出来的图片
    def cut_random(self, imgrgb, cut_rate=0.02):
        h, w, c = imgrgb.shape

        cut_h_u = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_h_d = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_w_l = random.choice(range(1, max(int(w * cut_rate), 2)))
        cut_w_r = random.choice(range(1, max(int(w * cut_rate), 2)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]
        return imgrgb

    # 模糊处理
    def img_blur(self, imgrgb):
        choice_list = [3, 5]

        my_choice = random.sample(choice_list, 1)
        img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
        return img_blur

    def __getitem__(self, index):

        img_path_list = self.all_batch_img_list[index]
        data_list = []
        label_list = []

        for img_path in img_path_list:
            # print(img_path)
            label_str = img_path.split('/')[-2]
            label = self.label_dict[label_str]
            img = None
            try:
                if self.phase == 'train':
                    img = cv2.imread(img_path)
                else:
                    img = Image.open(img_path)

            except IOError:
                print(f"Error: read {img_path} fail")

            # gray_img = img.convert('LA')
            # gray_img = PIL.ImageOps.equalize(gray_img, mask=None)
            # img = PIL.ImageOps.equalize(img, mask=None)
            # 添加随机裁剪
            if self.phase == 'train':
                # 随机模糊
                if random.random() > 1:
                    img = self.img_blur(img)
                try:
                    img = self.cut_random(img)
                except:
                    pass
                    # print(img_path)
                    # exit(0)
                # 将opencv读出来的图像转化为BGR
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

            img, _, _ = letterbox_image(img, self.input_shape[1:])
            img_tensor = self.transforms(img)
            data_list.append(img_tensor)
            label_list.append(label)

        return data_list, label_list

    def __len__(self):
        return len(self.all_batch_img_list)


class Dataset_reg(data.Dataset):
    def __init__(self, root, phase='train', input_shape=(3, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape
        self.sample_nums = 1000

        random.seed(777)

        data_list = []
        cls_idx = 0
        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            files = os.listdir(label_path)
            selects = random.sample(files, min(self.sample_nums, len(files)))
            for each in selects:
                if not each.endswith('.jpg'): continue
                if '副本' in each: continue
                # if 'Thumbs.db:encryptable' in img_list: img_list.remove('Thumbs.db:encryptable')
                img_path = os.path.join(label_path, each)
                data_list.append('%s %d' % (img_path, cls_idx))
            cls_idx += 1

        self.data_list = data_list
        self.num_classes = cls_idx

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(degrees=180),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    # 随机裁剪
    def cut_random(self, imgrgb, cut_rate=0.02):
        h, w, c = imgrgb.shape

        cut_h_u = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_h_d = random.choice(range(1, max(int(h * cut_rate), 2)))
        cut_w_l = random.choice(range(1, max(int(w * cut_rate), 2)))
        cut_w_r = random.choice(range(1, max(int(w * cut_rate), 2)))
        h_cut = random.choice(range(0, cut_h_d + cut_h_u))
        w_cut = random.choice(range(0, cut_w_l + cut_w_r))
        imgrgb = imgrgb[h_cut:h - cut_h_d - cut_h_u + h_cut, w_cut:w - cut_w_r - cut_w_l + w_cut, :]
        return imgrgb

    # 模糊处理
    def img_blur(self, imgrgb):
        choice_list = [3, 5]
        my_choice = random.sample(choice_list, 1)
        img_blur = cv2.blur(imgrgb, (my_choice[0], my_choice[0]))
        return img_blur

    # def MedianBlur(self, imgrbg):

    def __getitem__test(self, index):
        sample = self.data_list[index]
        splits = sample.split()
        img_path = splits[0]
        try:
            if self.phase == 'train':
                img = cv2.imread(img_path)
            else:
                img = Image.open(img_path)

        except IOError:
            print("Error: read %s fail" % img_path)

        # 添加随机裁剪
        if self.phase == 'train':
            # 随机模糊
            try:
                # equalize
                equ = A.Equalize(mode="cv", by_channels=True, mask=None, always_apply=False, p=1)
                img = equ(image=img)["image"]

                # random blur
                p1 = random.random()
                if p1 <= 0.33:
                    img = self.img_blur(img)
                elif p1 <=0.66:
                    blur = A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=1)
                    img = blur(image=img)["image"]

                # random noise
                p2 = random.random()
                if p2 <= 0.33:
                    iso_noise = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1)
                    img = iso_noise(image=img)["image"]
                elif p2 <=0.66:
                    g_noise = A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=1)
                    img = g_noise(image=img)["image"]

            except:
                print('img_path', img_path)
            # 将opencv读出来的图像转化为BGR
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

        img, _, _ = letterbox_image(img, self.input_shape[1:])

        data = self.transforms(img)
        label = np.int32(splits[1])

        return data.float(), label

    def __getitem__(self, index):
        sample = self.data_list[index]
        splits = sample.split()
        img_path = splits[0]
        try:
            if self.phase == 'train':
                img = cv2.imread(img_path)
            else:
                img = Image.open(img_path)

        except IOError:
            print("Error: read %s fail" % img_path)

        if self.phase == 'train':
            # 随机模糊
            try:
                if random.random() > 0.5:
                    # exit()
                    img = self.img_blur(img)
                img = self.cut_random(img)
            except:
                print('img_path', img_path)
            # 将opencv读出来的图像转化为BGR
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

        img, _, _ = letterbox_image(img, self.input_shape[1:])

        data = self.transforms(img)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.data_list)
