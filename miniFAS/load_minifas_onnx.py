import os
import shutil
import cv2
import cv2 as cv
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from load_llie_onnx import lowlight
warnings.filterwarnings('ignore')

model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_1

def predict_two_model(image):
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    # image = cv2.imread(img_path)
   
    image_bbox, conf = model_test.get_bbox(image)
    # print("image: ", img_path, "conf: ", conf)
    if conf < 0.7:
        # dict= '/home/linhhima/FAS_Thuan/unrecognize_face_smartphone/'
        # if not os.path.exists(dict):
        #     os.makedirs(dict) 
        # dest = os.path.join(dict, img_name)
        # shutil.copyfile(img_path, dest) 
        return "none"

    prediction = np.zeros((1, 3))
    # test_speed = 0
    # sum the prediction from single model's result
    for model in [model_1, model_2]:
        model_name = model.split("/")[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)[0]

        
        prediction += model_test.predict(img, model)
    print("prediction: ", prediction)
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    # print("value: ", value)
    return "real" if label == 1 else "fake"