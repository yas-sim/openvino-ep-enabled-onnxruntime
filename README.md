# OpenVINO Execution Provider Enabled `onnxruntime`

## 1. Description
ONNX runtime is a deep learning inferencing library developed and maintained by Microsoft.  
<a href=https://github.com/microsoft/onnxruntime>'microsoft/onnxruntime' on GitHub</a>  

ONNX runtime can load the ONNX format DL models and run it on wide variety of systems. It supports multiple processors, OSes, and programming languages.  
Also, ONNX runtime supports multiple `execution providers(EP)` which is the backend inferencing engine library. ONNX runtime supports **Intel OpenVINO toolkit**, Intel DNNL, Intel nGraph, nVIDIA TensorRT, DirectML, ARM compute library, Android neural networks API, and many more EPs.  

However, Intel OpenVINO EP is not enabled in the prebuilt binary distribution of ONNX runtime (v1.3.0).  

In this project, I built the ONNX runtime from the source code and enabled the OpenVINO execution provider.  
The project includes the steps to build and install ONNX runtime and a simple sample code to try ONNX runtime.  

**You can load the ONNX DL model directry in your program and run it as fast and efficient as OpenVINO with this OpenVINO EP.**  
The performance is OpenVINO >> ONNX runtime (C++) > ONNX runtime (Python) in general. The performance difference between OpenVINO and ONNX runtime is around 2ms to 10ms based on my test result (HW: Core i7-6770HQ).  

The sample code are simple CNN image classification program and the DL model is ResNet-50.  

----

ONNX runtimeはマイクロソフトが開発、保守しているディープラーニング推論ライブラリです。  
<a href=https://github.com/microsoft/onnxruntime>'microsoft/onnxruntime' on GitHub</a>  

ONNX runtimeはサポートする数多くのシステム上でONNXフォーマットのDLモデルをロードし実行することが可能です。多くのプロセッサ、OS、プログラミング言語をサポートします。  
またONNX runtimeは複数の`Execution provider(EP)`と呼ばれるバックエンド推論ライブラリをサポートします。**Intel OpenVINO toolkit**, Intel DNNL, Intel nGraph, nVIDIA TensorRT, DirectML, ARM compute library, Android neural networks APIなど数多くのEPをサポートしています。  

しかしながら提供されているビルド済みバイナリパッケージではOpenVINO EPがイネーブルされていません (v1.3.0)。

そこで、このプロジェクトではONNX runtimeをソースコードからビルドし、OpenVINO EPをイネーブルしてみました。  

**このOpenVINO EPにより、(MOでIRに変換することなく)プログラム内で直接ONNX DLモデルを読み込み、OpenVINOのように高速に効率よく推論を行うことが可能になります。**  
パフォーマンスはおおむねOpenVINO >> ONNX runtime (C++) > ONNX runtime (Python)の順になるようです。私のテスト結果によると、パフォーマンスの差は2ms~10ms程度でした。(HW:Core i7-6770HQ)  

サンプルプログラムは簡単な画像分類(CNN)のプログラムでResnet-50を使用しています。


## 2. Prerequisites
- **OpenVINO 2020.2**
  - ONNX runtime v1.3.0 is compatible with OpenVINO 2020.2.  
  - If you haven't installed it, go to the OpenVINO web page and follow the [*Get Started*](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) guide to do it.  


## 3. Build and Install ONNX runtime

Linux
```sh
# Build onnxruntime
git clone https://github.com/microsoft/onnxruntime
git checkout v1.3.0
cd onnxruntime
./build.sh --config Release --build_shared_lib --build_wheel --enable_pybind --use_openvino CPU_FP32 --skip_tests

#Install Python module (optional)
pip3 install ./build/Linux/Release/dist/onnxruntime_openvino-1.3.0-cp36-cp36m-linux_x86_64.whl
```

Windows
```sh
# Build onnxruntime
git clone https://github.com/microsoft/onnxruntime
git checkout v1.3.0
cd onnxruntime
build.amd64.1411.bat --config Release --cmake_generator "Visual Studio 16 2019" --build_shared_lib --build_wheel --enable_pybind --use_openvino CPU_FP32 --skip_tests

#Install Python module (optional)
pip install .\build\Windows\Release\Release\dist\onnxruntime_openvino-1.3.0-cp36-cp36m-win_amd64.whl
```

## 4. Build and Run the sample applications (C++ and Python)

Linux
```sh
# Download Resnet-50 model and class label text
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-7.onnx
wget https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt

#Build C++ sample app
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

#Run
cp $INTEL_OPENVINO_DIR/deployment_tools/demo/car.png .
cp build/onnxtest .
cp ../onnxruntime/build/Linux/Release/libonnxruntime.so .
cp ../onnxruntime/build/Linux/Release/libcustom_op_library.so .
./onnxtest
```

Windows
```sh
# Download Resnet-50 model and class label text
bitsadmin /transfer download https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v1-7.onnx %CD%\resnet50-v1-7.onnx
bitsadmin /transfer download https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt %CD%\synset_words.txt

#Build C++ sample app
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release ..
msbuild onnxtest.sln /p:Configuration=Release
cd ..

#Run
copy "%INTEL_OPENVINO_DIR%\deployment_tools\demo\car.png" .
copy build\Release\onnxtest.exe .
copy ..\onnxruntime\build\Windows\Release\Release\onnxruntime.dll .
copy ..\onnxruntime\build\Windows\Release\Release\custom_op_library.dll .
onnxtest.exe
```

## 5. Test Environment
Ubuntu 18.04 / Windows 10 1909
OpenVINO 2020.2
ONNX runtime 1.3.0

## See Also  
* [Using Open Model Zoo demos](../../README.md)  
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)  
* [Model Downloader](../../../tools/downloader/README.md)  
* [ONNX runtime](https://github.com/microsoft/onnxruntime)