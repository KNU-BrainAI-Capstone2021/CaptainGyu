from django.shortcuts import render
from .models import *
import numpy as np
from PIL import Image
import torch
import os
import io
from .utils import mtcnn_detect_img, detection
# Create your views here.

def show_main(request):
    if request.method == 'POST':
        print(request.FILES)
        targetImg_data = request.FILES['file']
        targetImg_data = targetImg_data.read()
        targetImg = Image.open(io.BytesIO(targetImg_data))
        
        #img = Image.open(io.BytesIO(targetImg_data)).resize()~
        #img = np.asarray(img) / 255.0
        targetImg.save('test.png')
        print(type(targetImg))
        try:
            targetImg = mtcnn_detect_img(targetImg)
            precision, result = detection(targetImg)
        except:
            print("no Human FAce!")
            precision, result = 'Face Not Found!', '???'


        return render(request, 'main/index.html', context={'precision':precision, 'result': result})

    elif request.method == 'GET':
        #form = ImageForm()

        return render(request, 'main/index.html', context={'precision':'???', 'result':'???'})