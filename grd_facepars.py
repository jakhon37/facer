

import numpy as np
import gradio as gr
import cv2
# from ultralytics import YOLO 
import mediapipe as mp   
import facer
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from  face_clean import FaceProcessor


mask_cls = FaceProcessor(debug=True)
face_attr = facer.face_attr("farl/celeba/224", device='cuda')
facer_detect = facer.face_detector("retinaface/mobilenet", device='cuda')
area_dict = {'background':0, 'neck':1, 'face':2, 'cloth':3, 'rr':4, 'lr':5, 'rb':6, 'lb':7, 're':8, 'le':9, 'nose':10, 'imouth':11, 'llip':12, 'ulip':13, 'hair':14, 'eyeg':15, 'hat':16, 'earr':17, 'neck_l':18}
# area_dict = {'background':'background', 'neck':'neck', 'face':'face', 'cloth':'cloth', 'rr':'right ear', 'lr':'left ear', 'rb':'right brown', 'lb':'left brown', 're':'right eye', 'le':'left eye', 'nose':'nose', 'imouth':'mouth', 'llip':'lower lip', 'ulip':'upper lip', 'hair':'hair', 'eyeg':'eyeg', 'hat':'hat', 'earr':'ear ring', 'neck_l':'necklace'}
area_dict_inf = {
    'background': 'background',
    'neck': 'neck',
    'face': 'face',
    'cloth': 'cloth',
    'right ear': 'rr',
    'left ear': 'lr',
    'right brown': 'rb',
    'left brown': 'lb',
    'right eye': 're',
    'left eye': 'le',
    'nose': 'nose',
    'mouth': 'imouth',
    'lower lip': 'llip',
    'upper lip': 'ulip',
    'hair': 'hair',
    'eyeg': 'eyeg',
    'hat': 'hat',
    'ear ring': 'earr',
    'necklace': 'neck_l'
}
areaList = ['background', 'neck', 'face', 'cloth', 'right ear', 'left ear', 'right brown', 'left brown', 'right eye', 'left eye', 'nose', 'mouth', 'lower lip', 'upper lip', 'hair', 'eyeg', 'hat', 'ear ring', 'necklace']

debug = True


def infr(img, area_list):
    # img = cv2.imread(img)
    if len(area_list)>0:
        for i in range(len(area_list)):
            area_list[i] = area_dict_inf[area_list[i]]
        print('|==== area list- ', area_list)
        g_area_list, mask_list, area_list = mask_cls.face_obj_mask2(img, area_list=area_list)
        # g_area_list , mask_list, area_list = g_area_list[0], mask_list[0], area_list

    # elif len(area_list)==1:
    #     print('|==== area list- ', area_list[0])
    #     g_area_list, mask_list, area_list = mask_cls.face_obj_mask2(img, area_name=area_dict_inf[area_list[0]])
    #     g_area_list , mask_list, area_list = g_area_list[0], mask_list[0], area_list
    else:
        g_area_list, mask_list, area_list = mask_cls.face_obj_mask2(img)

    return g_area_list , mask_list, area_list





# demo = gr.Interface(yolo_predict, gr.Image(shape=(200, 200)), "image")

demo = gr.Interface(
    fn=infr, 
    inputs=[
        gr.inputs.Image(),
        gr.inputs.CheckboxGroup(areaList, label="Area of interest")
            ],  # Input component
    outputs=[
        gr.outputs.Image(label="Output Image", type="numpy"), 
        gr.outputs.Image(label="Output mask", type="numpy"), 
        gr.outputs.Textbox(label="Gender"),
        # gr.outputs.Textbox(label="Hair style")
    ],  # Output components

    title="AIVAR HAIR CLASSIFICATION",
    description="This model detects gender and classifies hair styles, developed by AIVAR"
)


demo.launch(share=True)