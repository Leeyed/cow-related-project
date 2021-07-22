import os
import random
import math
import shutil
import yaml

def data_split_cleaned(old_path, new_path):
    if os.path.exists(new_path) == 0:
        print ("create new path")
        os.makedirs(new_path)

    random.seed(777)

    ### randomly split data into train set and test set. proportion = 8:2
    for root_dir, sub_dirs, file in os.walk(old_path):
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))
            file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))
            random.shuffle(file_names)
            for i in range(len(file_names)):
                if i < math.floor(0.8*len(file_names)):
                    sub_path = os.path.join(new_path, 'train_set', sub_dir)
                else:
                    sub_path = os.path.join(new_path, 'test_set', sub_dir)
                if os.path.exists(sub_path) == 0:
                    os.makedirs(sub_path)
                shutil.copy(os.path.join(root_dir, sub_dir, file_names[i]), os.path.join(sub_path, file_names[i]))

    ### delete the data < 24 pics
    for root_dir, sub_dirs, file in os.walk(os.path.join(new_path, 'train_set')):
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))
            if len(file_names) < 24:
                print("del ", sub_dir)
                if os.path.exists(os.path.join(new_path, 'test_set', sub_dir)) == 1:
                    shutil.rmtree(os.path.join(new_path, 'test_set', sub_dir))
                if os.path.exists(os.path.join(new_path, 'train_set', sub_dir)) == 1:
                    shutil.rmtree(os.path.join(new_path, 'train_set', sub_dir))
    print("split finished!")

def get_clear_picName(path, fileName):
    imgList = os.listdir(path)
    # if os.path.exists(fileName):
    with open(fileName, "w") as f:
        for imgpath in imgList:
            if imgpath.endswith('.jpg'):
                f.write("{}\n".format(imgpath))
    print("succeed! get clear pic Name and saved")


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    # get_clear_picName(config['clear_img_path'], config['clear_txt'])
    data_split_cleaned(config['old_img_path'], config['new_img_path'])
