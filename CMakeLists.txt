
cmake_minimum_required(VERSION 3.11)
project(CountObj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# CUDA 配置
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.3/bin/nvcc)
enable_language(CUDA)


# 添加src文件夹中的源文件和头文件
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB SRC_HPP "src/*.hpp")
file(GLOB SRC_CU "src/*.cu")  # 单独获取 CUDA 文件


# OpenCV
find_package(OpenCV REQUIRED)

# 包含目录配置
include_directories(
    /usr/local/cuda-12.3/include  # CUDA
    /opt/TensorRT/TensorRT-10.3.0.26/include/  # TensorRT
    ${CMAKE_CURRENT_SOURCE_DIR}/src  # 确保包含src文件夹
    ${EIGEN3_INCLUDE_DIR}  # 包含Eigen库的头文件路径
)

# 链接目录配置
link_directories(
    /usr/local/cuda-12.3/lib64
    /opt/TensorRT/TensorRT-10.10.0.31/lib/

    /usr/local/lib
)

add_executable(build
    main11.cpp
    ${SRC_CPP}
     ${SRC_CU}
)

# 设置 CUDA 架构
set_target_properties(build PROPERTIES 
    CUDA_ARCHITECTURES "61;70;75;89"
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接库配置
target_link_libraries(build PRIVATE
    # OpenCV
    ${OpenCV_LIBS} 
   
    # TensorRT
    nvinfer nvinfer_plugin nvonnxparser 
    
    # CUDA
    cudart
)

# 附加编译选项（可选）
target_compile_options(build PRIVATE
    -Wall
    -Wno-deprecated-declarations
    -O2
)
