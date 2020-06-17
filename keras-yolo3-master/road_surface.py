import sys
import argparse
import os
os.environ['KERAS_BACKEND']='tensorflow'
import cv2
import numpy as np

import matplotlib.pyplot as plt
from yolo import YOLO

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

yolo=YOLO()

# if __name__ == '__main__':

    
#     yolo = YOLO()   
#     for filename in os.listdir(path):        
#         image_path = path+'/'+filename
           
#         image = Image.open(image_path)
#         r_image = yolo.detect_image(image)
    
#         r_image.show() 

#     yolo.close_session()

def road_surface_detect(dictionary_name):

    for filename in os.listdir(r'./'+dictionary_name): #单双引号都行
        
        image_path=r'./'+dictionary_name+'/'+filename  #单双引号都行

        image=Image.open(image_path)

        r_image = yolo.detect_image(image)

        r_image.show()

    yolo.close_session()  


road_surface_detect('TextImages') #单双引号都行