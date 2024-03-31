import onnxruntime
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F 
import model_ZeroDCE
import os
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import argparse
import time
import model_ZeroDCE
import numpy as np
import torchvision 
from PIL import Image
import glob
import time
import PIL
from matplotlib import pyplot as plt
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def lowlight(data_lowlight):
    scale_factor = 12
    # data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)
	
    # print(data_lowlight.shape)
    torch_model = model_ZeroDCE.enhance_net_nopool(scale_factor)
    torch_model.load_state_dict(torch.load('/home/user/FAS/miniFAS/snapshots_Zero_DCE++/Epoch99.pth', map_location=torch.device('cpu')))
    torch_model.eval()

    # print('type(torch_model): ', type(torch_model))

    onnx_program = torch.onnx.dynamo_export(torch_model, data_lowlight)
    onnx_input = onnx_program.adapt_torch_inputs_to_onnx(data_lowlight)
    
    onnx_program.save("/home/user/FAS/miniFAS/ZeroDCE++1.onnx")

    ort_session = onnxruntime.InferenceSession("/home/user/FAS/miniFAS/ZeroDCE++1.onnx", providers=['CPUExecutionProvider'])
    
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)}
    # start = time.time()
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    # end_time = (time.time() - start)
    # print(type(onnxruntime_outputs))
    # print('onnxruntime_outputs[0]: ', onnxruntime_outputs[0].shape)
	
    output1 = onnxruntime_outputs[0].reshape(onnxruntime_outputs[0].shape[1], onnxruntime_outputs[0].shape[2], onnxruntime_outputs[0].shape[3])
    # output2 = onnxruntime_outputs[1].reshape(onnxruntime_outputs[1].shape[1], onnxruntime_outputs[1].shape[2], onnxruntime_outputs[1].shape[3])

    red_channel = output1[0]
    green_channel = output1[1]
    blue_channel = output1[2]
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    # result_path = './data/result_Test_Part2_onnx/'
	
    # result_path = os.path.join(result_path, image_name)
   
    # plt.imsave(result_path, rgb_image)
    return rgb_image

# if __name__ == '__main__':

# 	with torch.no_grad():

# 		filePath = '/home/user/FAS/miniFAS/dataset/Test_Part2'	
# 		file_list = os.listdir(filePath)
# 		sum_time = 0
# 		for file_name in file_list:
# 			print("file_name:",file_name)
# 			path_to_image = os.path.join(filePath, file_name)
# 			print("path_to_image:",path_to_image)
# 			sum_time = sum_time + lowlight(path_to_image)
# 		print(sum_time)
		