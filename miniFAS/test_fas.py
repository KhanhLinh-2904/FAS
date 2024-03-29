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
from check_lowlight import is_low_light
from load_llie_onnx import lowlight
warnings.filterwarnings('ignore')

model_test = AntiSpoofPredict(0)
dataset ="/home/linhhima/FAS/miniFAS/datasets/Test_Part2/"
 
# sample = "/home/linhhima/FAS_Thuan/datasets/real/0001_00_00_01_2.jpg"

model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_1

def predict_fas(img):
    prediction = model_test.predict(img,model)
    label = np.argmax(prediction)
    confident_score = prediction[0][label]
    if label == 1:
        # REAL 
        return "real"
    else:
        # 2D / 3D fake
        return "fake"
    
def predict_fas_face(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_bbox = model_test.get_bbox(img)
    
    model_name = model.split('/')[-1]
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    param = {
        "org_img": img,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True,
    }
    if scale is None:
        param["crop"] = False
    img = CropImage().crop(**param)
    
    prediction = model_test.predict(img, model)
    label = np.argmax(prediction)
    confident_score = prediction[0][label]
    if label == 1:
        # REAL 
        return "real"
    else:
        # 2D / 3D fake
        return "fake"
    

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
    
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0

    label = 'fake'
    images = os.listdir(dataset)
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        threshold = 100
        if is_low_light(img_path,threshold):
            img = lowlight(img_path)
        # try:
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