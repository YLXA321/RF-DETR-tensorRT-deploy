#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <memory>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::cerr << "CUDA error [" << static_cast<int>(err__) << "] "                \
                      << cudaGetErrorString(err__) << " at " << __FILE__                  \
                      << ":" << __LINE__ << std::endl;                                    \
            assert(false);                                                                \
        }                                                                                 \
    } while (0)
#endif

// // 管理 TensorRT/NV 对象：调用 p->destroy()
// // 注意：只有 TensorRT 对象使用此 helper（它们需要调用 destroy() 而不是 delete）。
// template<typename T>
// inline std::shared_ptr<T> make_nvshared(T* ptr){
//     return std::shared_ptr<T>(ptr, [](T* p){ if(p) p->destroy(); });
// }


// 管理 TensorRT/NV 对象：使用 delete p;
template<typename T>
inline std::shared_ptr<T> make_nvshared(T* ptr){
    return std::shared_ptr<T>(ptr, std::default_delete<T>());
}

/*-------------------------- YOLOV5_SEGMENT --------------------------*/
struct detectRes {
    int label { -1 };
    float confidence { 0.f };
    cv::Rect box {};
    cv::Scalar box_color {};
};


#endif // UTILS_HPP