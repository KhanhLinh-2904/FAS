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
from src.data_io import transform as trans
import onnxruntime
import torch.nn.functional as F
warnings.filterwarnings('ignore')

model_1 = 'resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
model_2 = 'resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth'
model = model_1
# device = torch.device("cuda:{}".format(0)
                                #    if torch.cuda.is_available() else "cpu")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_fas_onnx(model_path, image):
    torch_model = AntiSpoofPredict(0)
    image_cropper = CropImage()
   
    image_bbox, conf = torch_model.get_bbox(image)
    if conf < 0.7:
        # print("Have no face")
        return "none"
    prediction = np.zeros((1, 3))
    model_name = model_path.split("/")[-1]
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
    test_transform = trans.Compose([
        trans.ToTensor(),
    ])
    img = test_transform(img)
    img = img.unsqueeze(0).cpu()
    # print("type of img: ",img.shape)
    # print("model_path: ",model_path)

    torch_model_loaded = torch_model._load_model(model_path)
    torch_model_loaded.eval()

    # torch.onnx.export(torch_model_loaded.to('cpu'), img,'onnxfile.onnx')
    onnx_program = torch.onnx.dynamo_export(torch_model_loaded, img)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(img)
    
    model_name = model_name.split(".pth")[0]
    # print("model_name: ",model_name)
    # if model_name == '2.7_80x80_MiniFASNetV2':
    save_name = model_name +'.onnx'
    # print("save name: ",save_name)
    onnx_program.save(save_name)
    ort_session = onnxruntime.InferenceSession(save_name, providers=['CPUExecutionProvider'])
    onnx_input = {k.name: to_numpy(v) for k,v in zip(ort_session.get_inputs(), onnx_input)}
    onnxruntime_outputs = ort_session.run(None, onnx_input)
    onnxruntime_outputs = torch.Tensor(onnxruntime_outputs)
    onnxruntime_outputs = onnxruntime_outputs.view(1,3)
    onnxruntime_outputs = F.softmax(onnxruntime_outputs).cpu().numpy()
    # print(type(onnxruntime_outputs))
    # print(onnxruntime_outputs)
    return onnxruntime_outputs


# if __name__ == '__main__' :
#     with torch.no_grad():
#         filePath = '/home/user/FAS/miniFAS/dataset/Real/'
#         file_list = os.listdir(filePath)
#         for file_name in file_list:
#             path_to_image = os.path.join(filePath, file_name)
#             image = cv2.imread(path_to_image)
#             prediction = np.zeros((1, 3))
#             if image is None:
#                 print("No image")
#             # print(image.shape)
#             for model in [model_1, model_2]:
#                 prediction += load_fas_onnx(model, image)
#             label = np.argmax(prediction)
#             print("real\n")  if label == 1 else print("fake\n")
            