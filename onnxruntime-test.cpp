// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

// Original source code URL: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp

#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>

#include <chrono>
#include <opencv2/opencv.hpp>

//OPENVINO
#define USE_OPENVINO
//OPENVINO

#include <core/session/onnxruntime_cxx_api.h>

//OPENVINO
#include <core/providers/openvino/openvino_provider_factory.h>
//OPENVINO

int main(int argc, char* argv[]) {

	// Read class label text data
	std::ifstream label_file("synset_words.txt");  
	std::string str;
	std::vector<std::string> labels;
	while(getline(label_file, str)) labels.push_back(str);
	label_file.close();

	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

	// initialize session options if needed
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	// Sets graph optimization level
	// Available levels are
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible opitmizations

	//OPENVINO
	session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);  // Disable high level optimization by ONNX runtime
	//OPENVINO

	//OPENVINO
	std::string inference_device = "CPU_FP32";
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options, inference_device.c_str()));
	std::cout << "[INFO] Using OpenVINO execution provider" << std::endl;
	//OPENVINO

	//*************************************************************************
	// create session and load model into memory
#ifdef _WIN32
	const wchar_t* model_path = L"resnet18-v2-7.onnx";
#else
	const char* model_path = "resnet18-v2-7.onnx";
#endif

	printf("Using Onnxruntime C++ API\n");
	Ort::Session session(env, model_path, session_options);

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	Ort::AllocatorWithDefaultOptions allocator;

	// print number of model input nodes
	size_t num_input_nodes = session.GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
											// Otherwise need vector<vector<>>

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
	// print input node names
	char* input_name = session.GetInputName(i, allocator);
	printf("Input %d : name=%s\n", i, input_name);
	input_node_names[i] = input_name;

	// print input node types
	Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType type = tensor_info.GetElementType();
	printf("Input %d : type=%d\n", i, type);

	// print input shapes/dims
	input_node_dims = tensor_info.GetShape();
	printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
	for (int j = 0; j < input_node_dims.size(); j++)
		printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}

	//*************************************************************************
	// print model output layer (node names, types, shape etc.)
	//Ort::AllocatorWithDefaultOptions allocator;

	// print number of model output nodes
	size_t num_output_nodes = session.GetOutputCount();
	std::vector<const char*> output_node_names(num_output_nodes);
	std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 output node {1, 3, 224, 224}.
											// Otherwise need vector<vector<>>

	printf("Number of outputs = %zu\n", num_output_nodes);

	// iterate over all output nodes
	for (int i = 0; i < num_output_nodes; i++) {
	// print output node names
	char* output_name = session.GetOutputName(i, allocator);
	printf("Output %d : name=%s\n", i, output_name);
	output_node_names[i] = output_name;

	// print output node types
	Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

	ONNXTensorElementDataType type = tensor_info.GetElementType();
	printf("Output %d : type=%d\n", i, type);

	// print output shapes/dims
	output_node_dims = tensor_info.GetShape();
	printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
	for (int j = 0; j < output_node_dims.size(); j++)
		printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
	}

	//*************************************************************************
	// Score the model using sample data, and inspect values

	size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
												// use OrtGetTensorShapeElementCount() to get official size!

	std::vector<float> input_tensor_values(input_tensor_size);

	// Load an image and fill the input blob (0.0-1.0)
	cv::Mat image = cv::imread("car.png");
	cv::resize(image, image, cv::Size(224,224));
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	uint8_t *buf = image.data;
	for(int h=0; h<224; h++) {
		for(int w=0; w<224; w++) {
			input_tensor_values[0 * (224*224) + h*224 + w] = (float)(*buf++)/255.f;    
			input_tensor_values[1 * (224*224) + h*224 + w] = (float)(*buf++)/255.f;    
			input_tensor_values[2 * (224*224) + h*224 + w] = (float)(*buf++)/255.f;    
		}
	}

	// create input tensor object from data values
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());

	// Benchmarking (sync, latency)
	class std::vector<struct Ort::Value,class std::allocator<struct Ort::Value> > output_tensors;
	std::cout<<"Start inferencing"<<std::endl;
	auto startTime = std::chrono::system_clock::now();
	size_t niter = 100;
	for(size_t i=0; i<niter; i++) {
		output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
	}
	auto endTime = std::chrono::system_clock::now();
	auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	std::cout<<"Finished inferencing"<<std::endl;
	std::cout<<inference_device<<std::endl;
	std::cout<<execTime/niter<<"ms/inference"<<std::endl;

	// Get pointer to output tensor float values
	float* output = output_tensors.front().GetTensorMutableData<float>();
	std::cout << "\nresults\n------------------" << std::endl;
	std::vector<int> idx;
	for(int i=0; i<1000; i++) idx.push_back(i);
	std::sort(idx.begin(), idx.end(), [output](const int& left, const int& right) { return output[left]>output[right]; } );
	for (size_t id = 0; id < 5; ++id) {
		std::cout << id <<  " : " << idx[id] << " : " << 
		std::fixed<< std::setprecision(2) << output[idx[id]] << " " << labels[idx[id]] << std::endl;
	}
	printf("Done!\n");
	return 0;
}