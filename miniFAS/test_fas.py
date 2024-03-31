import os
import shutil
import cv2
import cv2 as cv
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm
import torch
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from check_lowlight import is_low_light
from load_llie_onnx import lowlight
warnings.filterwarnings('ignore')
from load_minifas_onnx import load_fas_onnx
model_test = AntiSpoofPredict(0)
dataset ="/home/user/FAS/miniFAS/dataset/Real/"
 
# sample = "/home/linhhima/FAS_Thuan/datasets/real/0001_00_00_01_2.jpg"

model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_1

def predict_two_model(image):
    prediction = np.zeros((1, 3))
    for model in [model_1, model_2]:
        flag = load_fas_onnx(model, image)
        if type(flag) == type("none"):
            return "none"
        else:
            prediction += flag
    print("prediction: ", prediction)
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    print("value: ", value)
    return "real" if label == 1 else "fake"
    
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    label = 'real'
    images = os.listdir(dataset)
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        threshold = 100
        if is_low_light(img,threshold):
            print("Yes! Need to llie\n")
            img = lowlight(img)
        # try:
        cv2.imshow(image,img )
        prediction = predict_two_model(img)
        if prediction == "none":
            print("There is no face here!")
            continue
        # if prediction != label:
        #     print( image)
        #     directory= '/home/linhhima/FAS_Thuan/db_smartphone_fail'
        #     if not os.path.exists(directory):
        #         os.makedirs(directory) 
        #     destination = os.path.join(directory, image)
        #     shutil.copyfile(img_path, destination) 
        # except Exception as e: 
        #     print(e)
        #     continue
        # print(time.time() - tic)
    
        if prediction == "fake" and label == 'fake':
            tp += 1
        elif prediction == 'real' and label == 'fake':
            fn += 1
        elif prediction == 'fake' and label == 'real':
            fp += 1
        elif prediction == 'real' and label == 'real':
            tn += 1
            
    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)