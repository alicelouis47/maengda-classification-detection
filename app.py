import json
import ast
import pandas as pd
import os.path as op
from pathlib import Path
from PIL import Image
from icevision.models import *
from icevision.all import *
import icedata
import PIL, requests
import torch
from torchvision import transforms
import gradio as gr
from fastbook import *
from fastai.vision.all import *
import os.path as op
import pandas as pd
from tqdm import tqdm
from time import sleep
from glob import glob
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, plot_confusion_matrix ,  confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
import torch
import torchvision.transforms.functional as fn
from PIL import Image
from torchvision.transforms.functional import crop
import cv2
import numpy as np
import streamlit as st


st.title('Mangda detectio and classification')
st.header('AI Builders ปีที่ 2')
st.subheader('จัดทำโดย นายภัคพล อาจบุราย ชั้น ม.6 โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย บุรีรัมย์')
st.subheader('โพรเจกต์นี้ เป็นส่วนหนึ่งของโครงการ AI Builders (ก่อตั้งจากความร่วมมือของ VISTEC, AIResearch และ Central Digital)')
st.caption('Medium: shorturl.at/frCOR')
st.caption('GitHub: https://github.com/alicelouis47/mangda-detection')
st.caption('แบบสอบถามเพื่อประเมินคุณภาพของโมเดล: https://forms.gle/zqo2cCg3yepf4XKi6')
st.caption('หมายเหตุ: Carcinoscorpius_rotundicauda คือ แมงดาถ้วย(มีพิษ) Tachypleus_gigas คือ แมงดาจาน(กินได้)')


uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")
img_file_buffer = st.camera_input("Take a picture")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='input')
    # โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    model_loaded = model_from_checkpoint("./model_OBJ/mangda_det_660.pth")
    model = model_loaded['model']
    model_type = model_loaded["model_type"]
    backbone = model_loaded["backbone"]
    class_map = model_loaded["class_map"]
    img_size = model_loaded["img_size"]
    #โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    learn_BF = load_learner('./model_back_front/VGG16_fastai.pkl')
    learn_B = load_learner('./model_back/densenet201_fastai.pkl')
    learn_F = load_learner('./model_front/resnext50_fastai.pkl')
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad((img_size,img_size)), tfms.A.Normalize()])
    #predict image
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.7)
        # print("ภาพที่ %.2f" %num_img)
    img_out = img
    img_out = np.array(img_out)

    for i in range(len(pred_dict['detection']['bboxes'])):
        pre = pred_dict['detection']['bboxes'][i]

        BBox_tensor = pre.to_tensor()
        BBox_list= BBox_tensor.tolist()
        # Setting the points for cropped image
        left = int(BBox_list[0])
        top = int(BBox_list[1])
        right = int(BBox_list[2])
        bottom = int(BBox_list[3])
        # crop image
        im1 = img.crop((left, top, right, bottom))
        
        # im1.show()
        im1.resize((224,224))
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        im1 = trans(trans1(im1))
        im2 = np.array(im1)
        Pre_BF = learn_BF.predict(im2)[0]
        if Pre_BF == 'front' :# predict back or front
            Pre_F = learn_F.predict(im2)
            prop_F = Pre_F[2].max()
            prop_float = prop_F.item()

            st.metric(label='img'+str(i+1), value=Pre_F[0], delta=str(pre))
            # print('class:',Pre_F[0], ",accuracy =", '%.3f' %prop_float)
            # Draw draw boxes
            xmin = int(BBox_list[0])
            ymin = int(BBox_list[1])
            xmax = int(BBox_list[2])
            ymax = int(BBox_list[3])
            cv2.rectangle(img_out, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
            cv2.putText(img_out, 
                        Pre_F[0], 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * int(pred_dict['height']), 
                        (255,0,0), 2)
        else:
            Pre_B = learn_B.predict(im2)
            prop_B = Pre_B[2].max()
            prop_float = prop_B.item()
            st.metric(label='ภาพที่ 1'+str(i+1), value=Pre_B[0], delta=str(pre))

                # print('class:',Pre_B[0], ",accuracy =", '%.3f' %prop_float)
                # Draw draw boxes
            xmin = int(BBox_list[0])
            ymin = int(BBox_list[1])
            xmax = int(BBox_list[2])
            ymax = int(BBox_list[3])
            cv2.rectangle(img_out, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
            cv2.putText(img_out, 
                        Pre_B[0], 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * int(pred_dict['height']), 
                        (255,0,0), 2)

                
    img_out = Image.fromarray(img_out)
    st.image(img_out, caption='prediction')
        # display(img_out)
        # img_out = img_out.save("prediction " + str(num_img) + '.png')

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    st.image(img, caption='input')
    # โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    model_loaded = model_from_checkpoint("./model_OBJ/mangda_det_660.pth")
    model = model_loaded['model']
    model_type = model_loaded["model_type"]
    backbone = model_loaded["backbone"]
    class_map = model_loaded["class_map"]
    img_size = model_loaded["img_size"]
    #โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    learn_BF = load_learner('./model_back_front/VGG16_fastai.pkl')
    learn_B = load_learner('./model_back/densenet201_fastai.pkl')
    learn_F = load_learner('./model_front/resnext50_fastai.pkl')
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad((img_size,img_size)), tfms.A.Normalize()])
    #predict image
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.7)
        # print("ภาพที่ %.2f" %num_img)
    img_out = img
    img_out = np.array(img_out)

    for i in range(len(pred_dict['detection']['bboxes'])):
        pre = pred_dict['detection']['bboxes'][i]

        BBox_tensor = pre.to_tensor()
        BBox_list= BBox_tensor.tolist()
        # Setting the points for cropped image
        left = int(BBox_list[0])
        top = int(BBox_list[1])
        right = int(BBox_list[2])
        bottom = int(BBox_list[3])
        # crop image
        im1 = img.crop((left, top, right, bottom))
        
        # im1.show()
        im1.resize((224,224))
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        im1 = trans(trans1(im1))
        im2 = np.array(im1)
        Pre_BF = learn_BF.predict(im2)[0]
        if Pre_BF == 'front' :# predict back or front
            Pre_F = learn_F.predict(im2)
            prop_F = Pre_F[2].max()
            prop_float = prop_F.item()
            st.metric(label='img'+str(i+1), value=Pre_F[0], delta=str(pre))

            # print('class:',Pre_F[0], ",accuracy =", '%.3f' %prop_float)

            # Draw draw boxes
            xmin = int(BBox_list[0])
            ymin = int(BBox_list[1])
            xmax = int(BBox_list[2])
            ymax = int(BBox_list[3])
            cv2.rectangle(img_out, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
            cv2.putText(img_out, 
                        Pre_F[0], 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * int(pred_dict['height']), 
                        (255,0,0), 2)
        else:
            Pre_B = learn_B.predict(im2)
            prop_B = Pre_B[2].max()
            prop_float = prop_B.item()
            st.metric(label='ภาพที่ 1'+str(i+1), value=Pre_B[0], delta=str(pre))

                # print('class:',Pre_B[0], ",accuracy =", '%.3f' %prop_float)
                # Draw draw boxes
            xmin = int(BBox_list[0])
            ymin = int(BBox_list[1])
            xmax = int(BBox_list[2])
            ymax = int(BBox_list[3])
            cv2.rectangle(img_out, (xmin,ymin), (xmax,ymax), (255,0,0), 3)
            cv2.putText(img_out, 
                        Pre_B[0], 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * int(pred_dict['height']), 
                        (255,0,0), 2)

                
    img_out = Image.fromarray(img_out)
    st.image(img_out, caption='prediction')