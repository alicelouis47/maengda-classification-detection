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


st.title('Mangda classification')
st.header('AI Builders ปีที่ 2')
st.caption('')
st.subheader('จัดทำโดย นายภัคพล อาจบุราย ชั้น ม.6 โรงเรียนวิทยาศาสตร์จุฬาภรณราชวิทยาลัย บุรีรัมย์')
st.caption('')
st.subheader('โพรเจกต์นี้ เป็นส่วนหนึ่งของโครงการ AI Builders (ก่อตั้งจากความร่วมมือของ VISTEC, AIResearch และ Central Digital)')
st.caption('')
st.caption('Medium: shorturl.at/frCOR')
st.caption('')
st.caption('GitHub: https://github.com/alicelouis47/mangda-detection')
st.caption('แบบสอบถามเพื่อประเมินคุณภาพของโมเดล: https://forms.gle/zqo2cCg3yepf4XKi6')
st.caption('หมายเหตุ: Carcinoscorpius_rotundicauda คือ แมงดาถ้วย(มีพิษ) Tachypleus_gigas คือ แมงดาจาน(กินได้)')
st.caption('')
st.caption('')
st.caption('ผลกระทบจากการรับประทานแมงดาถ้วย')
st.caption('')
st.caption('จากข้อมูลทางวิชาการพบว่า สารพิษที่พบอยู่ในเนื้อและไข่ของแมงดาถ้วยคือ สารเทโทรโดท็อกซิน (tetrodotoxin) และซาซิท็อกซิน (saxitoxin)')
st.caption('ซึ่งเป็นสารชนิดเดียวกันกับที่พบในปลาปักเป้า เป็นสารที่ส่งผลต่อระบบควบคุมการหายใจถึงขั้นเสียชีวิต')
st.caption('โดยพิษในแมงดาถ้วยเกิดจาก 2 สาเหตุหลักๆ คือ')
st.caption('1. แมงดากินแพลงก์ตอนที่มีพิษ หรือกินสัตว์ทะเลอื่น ๆ ที่กินแพลงก์ตอนพิษเข้าไปทำให้สารพิษไปสะสมอยู่ในเนื้อและไข่ของแมงดา')
st.caption('2. เกิดจากแบคทีเรียในลำไส้ที่สร้างพิษขึ้นมาได้เอง หรือสาเหตุประกอบกันทั้งสอง และที่สำคัญสารพิษทั้ง 2 ชนิดนี้เป็นสารที่ทนต่อความร้อนได้ดี การปรุงอาหารด้วยความร้อนวิธีต่างๆ เช่น ต้ม ทอด หรือ อบ เป็นเวลานานมากกว่าชั่วโมงไม่สามารถทำลายสารพิษชนิดนี้ได้ ประชาชนจึงไม่ควรนำมาบริโภคอย่างเด็ดขาด')
st.caption('')
st.caption('')
st.caption('')
st.caption('https://www4.fisheries.go.th/local/index.php/main/view_activities/1210/88780')


uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    model_loaded = model_from_checkpoint("./model_OBJ/mangda_det_660.pth")
    model = model_loaded['model']
    model_type = model_loaded["model_type"]
    backbone = model_loaded["backbone"]
    class_map = model_loaded["class_map"]
    img_size = model_loaded["img_size"]
    # โหลดโมเดลที่เซฟมาเพื่อทำนายผลใน test set
    learn_BF = load_learner('./model_back_front/VGG16_fastai.pkl')
    learn_B = load_learner('./model_back/densenet201_fastai.pkl')
    learn_F = load_learner('./model_front/resnext50_fastai.pkl')
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad((img_size,img_size)), tfms.A.Normalize()])
    # predict image
    pred_dict  = model_type.end2end_detect(img, valid_tfms, model, class_map=class_map, detection_threshold=0.7)
    # print("ภาพที่ %.2f" %num_img)
    img_out = img
    img_out = np.array(img_out)

    # im1.show()
    img.resize((224,224))
    trans = transforms.ToPILImage()
    trans1 = transforms.ToTensor()
    im1 = trans(trans1(img))
    im2 = np.array(im1)
    Pre_BF = learn_BF.predict(im2)[0]
    if Pre_BF == 'front' :# predict back or front
        Pre_F = learn_F.predict(im2)
        prop_F = Pre_F[2].max()
        prop_float = prop_F.item()
        st.success(f"This is {Pre_F[0]}  with the probability of {prop_float*100:.02f}%")
        # print('class:',Pre_F[0], ",accuracy =", '%.3f' %prop_float)
        # Draw draw boxes
        st.image(img_out, use_column_width=True)
    
    else:
        Pre_B = learn_B.predict(im2)
        prop_B = Pre_B[2].max()
        prop_float = prop_B.item()
        st.success(f"This is {Pre_B[0]}  with the probability of {prop_float*100:.02f}%")
        # print('class:',Pre_B[0], ",accuracy =", '%.3f' %prop_float)
        # Draw draw boxes
        st.image(img_out, use_column_width=True)

         