# coding: utf-8
# This file contains the parameter used in train.py

from __future__ import division, print_function

from modules.misc_utils import parse_anchors, read_class_names
import math
import os
import yaml

TASK_PY_PATH = os.path.split(os.path.realpath(__file__))[0]
config_path = os.path.join(TASK_PY_PATH, 'cfg_body_bodai.yaml')
with open(config_path, 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

anchor_path = config['body_detect_anchor_path']
class_name_path = config['body_detect_class_path']
restore_path = config['body_detect_model_path']
new_size = config['new_size']
anchors = parse_anchors(anchor_path)

classes = read_class_names(class_name_path)
num_class = len(classes)


