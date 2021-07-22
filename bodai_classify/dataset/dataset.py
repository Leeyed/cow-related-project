import os
import sys
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from PIL import Image
import random
from utils import letterbox_image

sys.path.append("..")
torch.multiprocessing.set_sharing_strategy('file_system')


class BoDaiDataset(data.Dataset):
    def __init__(self, root, class_txt, phase='train', input_shape=(3, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape
        self.sample_nums = 1000
        train_val_scale = 0.2

        random.seed(777)
        txt_data = open(class_txt, 'w')

        data_list = []  # 所有的符合的数据集合
        train_list = []
        val_list = []
        cls_idx = 0
        for insert in os.listdir(root):
            # print(insert, ' ', cls_idx)
            if '+' in insert: continue
            insert_path = os.path.join(root, insert)
            files = os.listdir(insert_path)
            if len(files) > self.sample_nums:
                selects = random.sample(files, self.sample_nums)
            else:
                selects = random.sample(files, len(files))
            split_num = len(selects)*train_val_scale
            for i, each in enumerate(selects):
                if 'Thumbs.db' in each: continue
                img_path = os.path.join(insert_path, each)
                data_list.append('%s %d' % (img_path, cls_idx))
                if i <= split_num:
                    val_list.append('%s %d' % (img_path, cls_idx))
                else:
                    train_list.append('%s %d' % (img_path, cls_idx))

            cls_idx += 1
            # 写入class label
            txt_data.write(insert+'\n')
        txt_data.close()

        if self.phase == 'train':
            self.data_list = train_list
        else:
            self.data_list = val_list

        self.num_classes = cls_idx

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                # T.RandomRotation(degrees=180, resample=Image.BICUBIC, fill=(128,128,128)),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean, std)
            ])
        else:
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean, std)
            ])

    def __getitem__(self, index):
        sample = self.data_list[index]
        splits = sample.split()
        img_path = splits[0]
        try:
            img = Image.open(img_path)
        except IOError:
            print("Error: read %s fail" % img_path)

        img, _, _ = letterbox_image(img, self.input_shape[1:])

        data = self.transforms(img)
        # data = self.img2tensor(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.data_list)

