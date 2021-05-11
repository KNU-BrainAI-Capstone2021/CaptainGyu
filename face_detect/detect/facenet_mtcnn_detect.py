import cv2, mmcv
import facenet_pytorch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import torch
import time
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

cap = cv2.VideoCapture('./input/test.mp4')
est_FPS = 0

face_detector = MTCNN(keep_all=True, device=device)

video = mmcv.VideoReader('./input/test.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')
    preTime = time.time()
    # Detect faces
    boxes, _ = face_detector.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    try:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        
        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 480), Image.BILINEAR))
    except TypeError:
        continue
    dstTime = time.time()
    est_FPS = (1./(dstTime - preTime))
    print('\nfacenet_pytorch FPS : {}'.format(est_FPS))
print('\nDone')


dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim,isColor=True)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()