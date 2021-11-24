import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import os
import cv2
import numpy as np
import torch
from torch import nn, einsum
from sklearn.metrics import plot_confusion_matrix

#from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .cross_efficient_vit import CrossEfficientViT
#from utils import transform_frame
import glob
#from os import cpu_count
#import json
#from multiprocessing.pool import Pool
#from progress.bar import Bar
#import pandas as pd
#from tqdm import tqdm
#from multiprocessing import Manager
#from utils import custom_round, custom_video_round
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
from .transforms.albu import IsotropicResize
import yaml
import argparse

import facenet_pytorch
import cv2, mmcv
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import time

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])

def mtcnn_detect_img(img):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    face_detector = MTCNN(keep_all=True, device=device)

    box, _ = face_detector.detect(img)
    box = box[0]
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    try:
        # draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        margin = 20
        box[0] -= margin
        box[1] -= margin
        box[2] += margin
        box[3] += margin

        area = (box[0], box[1], box[2], box[3])
        img = img.crop(area)

    except TypeError:
        print("TypeError!!!")

    return img

def detection(img):
    model_path = './main/pretrained_models/cross_efficient_vit.pth'
    conf = './main/architecture.yaml'

    print(os.getcwd())
    with open(conf, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if os.path.exists(model_path):
        model = CrossEfficientViT(config=config)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
            model = model.cuda()
        model.eval()

    else:
        print("No model found.")

    transform = create_base_transform(config['model']['image-size'])
    numpy_image = np.array(img)
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    image = transform(image=img)['image']

    faces = torch.tensor(np.asarray(image))
    faces = torch.unsqueeze(faces, 0)
    faces = np.transpose(faces, (0, 3, 1, 2))
    if device == 'cpu':
        faces = faces.float()
    else:
        faces = faces.cuda().float()

    pred = model(faces)

    threshold = 0.55

    for idx, p in enumerate(pred):
        precision = torch.sigmoid(p).item()
        precision = round(precision, 4)
        if precision > threshold:
            detection_result = "FAKE"
        else:
            detection_rersult = "REAL"
        print(f'precision:{precision}, result:{detection_result}')


    return precision, detection_result
