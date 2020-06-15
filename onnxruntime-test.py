import sys
import time
import argparse

import cv2
import numpy as np
import onnxruntime

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, required=False, default='resnet18-v2-7.onnx', help='ONNX model file name')
parser.add_argument('-d', type=str, required=False, default='CPU_FP32', help='OpenVINO device name')
args = parser.parse_args()

print('model: ', args.m)
print('device: ', args.d)

available_devices = ['CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16', 'VAD-F_FP16']
if not args.d in available_devices:
    print('Device must be one of followings : ', available_devices)
    sys.exit(0)

print(onnxruntime.get_all_providers())
print(onnxruntime.get_device())

label = open('synset_words.txt').readlines()

# Available device names: CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16, VAD-F_FP32 
# VAD == Vision Accelerator Design == HDDL
options = onnxruntime.SessionOptions()
options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
onnxruntime.capi._pybind_state.set_openvino_device(args.d)

sess = onnxruntime.InferenceSession(args.m, options)

input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)

output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)  
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)

img = cv2.imread('car.png')
img = cv2.resize(img, ((input_shape[3], input_shape[2])))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2,0,1))
x = img.reshape(input_shape).astype(np.float32)/255.
print(x.shape)

iter=10
start = time.monotonic()
for i in range(iter):
    result = sess.run([output_name], {input_name: x})
end = time.monotonic()
print(((end-start)/iter)*1000,'ms/inference')

result=np.array(result).reshape((1000,))
idx = np.argsort(result)[::-1]
for i in range(5):
    print(idx[i]+1, result[idx[i]], label[idx[i]][:-1])