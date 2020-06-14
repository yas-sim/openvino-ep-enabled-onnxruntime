# Sample CMakeLists.txt file for an OpenVINO Inference Engine project
cmake_minimum_required (VERSION 2.8.1)

project(onnxtest)
set(TARGET_NAME onnxtest)     # name of executable file
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIE")

find_package(InferenceEngine 1.1 REQUIRED)
find_package(OpenCV REQUIRED)
add_definitions(-DUSE_OPENCV)

set(ONNXRT_DIR "../onnxruntime")
set(ONNXRT_INC_DIR "${ONNXRT_DIR}/include/onnxruntime;${ONNXRT_DIR}/include/onnxruntime/core/session;${ONNXRT_DIR}/onnxruntime")

if(UNIX)
  set(ONNXRT_LIB_DIR "${ONNXRT_DIR}/build/Linux/Release")
  set(ONNXRT_LIBS "onnxruntime;custom_op_library")
elseif(WIN32)
  set(ONNXRT_LIB_DIR "${ONNXRT_DIR}/build/Windows/Release/Release")
  set(ONNXRT_LIBS "onnxruntime;custom_op_library")
endif(UNIX)

include_directories( ${InferenceEngine_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${ONNXRT_INC_DIR} )
link_directories( ${ONNXRT_LIB_DIR} )

add_executable( ${TARGET_NAME} onnxruntime-test.cpp )    # list of source file(s)
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${ONNXRT_LIBS})
