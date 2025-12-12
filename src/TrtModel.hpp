#ifndef TRTMODEL_HPP
#define TRTMODEL_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "utils.hpp"

class TRTLogger : public nvinfer1::ILogger {
public:
    // 设置日志级别
    void setReportableSeverity(nvinfer1::ILogger::Severity severity) noexcept;

    // 获取单例实例
    static TRTLogger& getInstance();

protected:
    // 默认构造函数
    TRTLogger() = default;
    // 防止复制
    TRTLogger(const TRTLogger&) = delete;
    TRTLogger& operator=(const TRTLogger&) = delete;

private:
    // 日志级别
    nvinfer1::ILogger::Severity reportable_severity = nvinfer1::ILogger::Severity::kWARNING;

    // 获取当前时间字符串，格式如: "10/19/2025-14:45:30.123"
    std::string getCurrentTimeStr() const;

    // 实现 ILogger 接口的方法
    void log(Severity severity, const char* msg) noexcept override;
};


struct Config {
    int resolution {};
    std::array<float, 3> means{0.485f, 0.456f, 0.406f};
    std::array<float, 3> stds{0.229f, 0.224f, 0.225f};
};

struct TmpDet { int q; int cls; float score; };


class TrtModel
{

public:
    TrtModel(std::string onnxfilepath, bool fp16);
    ~TrtModel();

    std::vector<detectRes> postprocess(cv::Mat& frame);
    bool drawResult(cv::Mat& img, const std::vector<detectRes>& result);

private:
    bool genEngine();
    std::vector<unsigned char> load_engine_file();
    bool Runtime();
    bool trtIOMemory();
    void preprocess(const cv::Mat& image);

    bool createCudaGraph();
    size_t volume(const nvinfer1::Dims& d);

    // TRT runtime objects (shared_ptr with custom deleter)
    std::shared_ptr<nvinfer1::IRuntime> m_runtime{nullptr};
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine{nullptr};
    std::shared_ptr<nvinfer1::IExecutionContext> m_context{nullptr};
    cudaStream_t m_stream{nullptr};

    // Names
    std::string m_inputName;
    std::vector<std::string> m_outputNames; // can be 1 or 2 outputs

    // Host / device buffers
    float* m_input_device_memory{nullptr};
    float* m_input_host_memory{nullptr};

    // index 0 -> detection (host/device), index 1 -> prototype/segmentation (host/device)
    std::array<float*, 2> m_output_host_memory{{nullptr, nullptr}};
    std::array<float*, 2> m_output_device_memory{{nullptr, nullptr}};

    nvinfer1::Dims m_inputDims{}; // [1,3,896,896]
    nvinfer1::Dims m_detectDims{}; // [1,300,4] -->(x,y,w,h)
    nvinfer1::Dims m_labelDims{}; // [1,300,91] -->(n类别数+1背景)

    std::string m_enginePath;
    std::string onnx_file_path;
    bool FP16{false};           // 这里不能使用FP16量化，对检测的精度影响非常大

    int m_inputSize{0};
    int m_imgArea{0};
    int m_detectSize{0};
    int m_labelSize{0};

    // int kInputH{0};
    // int kInputW{0};

    Config config_;

    float kConfThresh = 0.5f;

    cudaGraphExec_t m_graphExec {nullptr};
    bool m_useGraph {false};
 
    const std::vector<std::string> CLASS_NAMES = {  /*需要检测的目标类别*/   
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse","sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis","snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove","skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich","orange", "broccoli", "carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv","laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

    const std::vector<std::vector<unsigned int>> COLORS_HEX = {     /*对不同的检测目标绘制不同的颜色*/
    {0x00, 0x72, 0xBD}, {0xD9, 0x53, 0x19}, {0xED, 0xB1, 0x20}, {0x7E, 0x2F, 0x8E}, {0x77, 0xAC, 0x30}, {0x4D, 0xBE, 0xEE},
    {0xA2, 0x14, 0x2F}, {0x4C, 0x4C, 0x4C}, {0x99, 0x99, 0x99}, {0xFF, 0x00, 0x00}, {0xFF, 0x80, 0x00}, {0xBF, 0xBF, 0x00},
    {0x00, 0xFF, 0x00}, {0x00, 0x00, 0xFF}, {0xAA, 0x00, 0xFF}, {0x55, 0x55, 0x00}, {0x55, 0xAA, 0x00}, {0x55, 0xFF, 0x00},
    {0xAA, 0x55, 0x00}, {0xAA, 0xAA, 0x00}, {0xAA, 0xFF, 0x00}, {0xFF, 0x55, 0x00}, {0xFF, 0xAA, 0x00}, {0xFF, 0xFF, 0x00},
    {0x00, 0x55, 0x80}, {0x00, 0xAA, 0x80}, {0x00, 0xFF, 0x80}, {0x55, 0x00, 0x80}, {0x55, 0x55, 0x80}, {0x55, 0xAA, 0x80},
    {0x55, 0xFF, 0x80}, {0xAA, 0x00, 0x80}, {0xAA, 0x55, 0x80}, {0xAA, 0xAA, 0x80}, {0xAA, 0xFF, 0x80}, {0xFF, 0x00, 0x80},
    {0xFF, 0x55, 0x80}, {0xFF, 0xAA, 0x80}, {0xFF, 0xFF, 0x80}, {0x00, 0x55, 0xFF}, {0x00, 0xAA, 0xFF}, {0x00, 0xFF, 0xFF},
    {0x55, 0x00, 0xFF}, {0x55, 0x55, 0xFF}, {0x55, 0xAA, 0xFF}, {0x55, 0xFF, 0xFF}, {0xAA, 0x00, 0xFF}, {0xAA, 0x55, 0xFF},
    {0xAA, 0xAA, 0xFF}, {0xAA, 0xFF, 0xFF}, {0xFF, 0x00, 0xFF}, {0xFF, 0x55, 0xFF}, {0xFF, 0xAA, 0xFF}, {0x55, 0x00, 0x00},
    {0x80, 0x00, 0x00}, {0xAA, 0x00, 0x00}, {0xD4, 0x00, 0x00}, {0xFF, 0x00, 0x00}, {0x00, 0x2B, 0x00}, {0x00, 0x55, 0x00},
    {0x00, 0x80, 0x00}, {0x00, 0xAA, 0x00}, {0x00, 0xD4, 0x00}, {0x00, 0xFF, 0x00}, {0x00, 0x00, 0x2B}, {0x00, 0x00, 0x55},
    {0x00, 0x00, 0x80}, {0x00, 0x00, 0xAA}, {0x00, 0x00, 0xD4}, {0x00, 0x00, 0xFF}, {0x00, 0x00, 0x00}, {0x24, 0x24, 0x24},
    {0x49, 0x49, 0x49}, {0x6D, 0x6D, 0x6D}, {0x92, 0x92, 0x92}, {0xB6, 0xB6, 0xB6}, {0xDB, 0xDB, 0xDB}, {0x00, 0x72, 0xBD},
    {0x50, 0xB7, 0xBD}, {0x80, 0x80, 0x00}};

};

#endif    // TRTMODEL_HPP