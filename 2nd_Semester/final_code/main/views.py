from django.shortcuts import render
from .models import *
import numpy as np
from PIL import Image
import torch
import os
import io
from .utils import mtcnn_detect_img, mtcnn_detect_video, detection, detection_video
import mimetypes

# Create your views here.

def show_main(request):
    if request.method == 'POST':
        print(request.FILES)
        targetImg_data_handler = request.FILES['file']
        print(targetImg_data_handler.temporary_file_path())
        upload_dtype = targetImg_data_handler.content_type.split('/')[0]
        targetImg_data = targetImg_data_handler.read()


        #img = Image.open(io.BytesIO(targetImg_data)).resize()~
        #img = np.asarray(img) / 255.0
        #targetImg.save('test.png')
        #print(type(targetImg))

        if upload_dtype == 'image':
            targetImg = Image.open(io.BytesIO(targetImg_data))
            try:
                targetImg = mtcnn_detect_img(targetImg)
                precision, result = detection(targetImg)
            except:
                print("no Human FAce!")
                precision, result = 'Face Not Found!', '???'

        elif upload_dtype == 'video':
            try:
                frames_tracked, img_pred = mtcnn_detect_video(targetImg_data_handler.temporary_file_path())
                precision, result = detection_video(frames_tracked, img_pred)
            except:
                print("no Human FAce!")
                precision, result = 'Face Not Found!', '???'

        else:
            precision, result = 'Only Image or video!', 'Check your file'

        targetImg_data_handler.close()
        return render(request, 'main/index.html', context={'precision':precision, 'result': result})

    elif request.method == 'GET':
        #form = ImageForm()

        return render(request, 'main/index.html', context={'precision':'???', 'result':'???'})