#include "TrtModel.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <sys/stat.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <random>


void TRTLogger::setReportableSeverity(nvinfer1::ILogger::Severity severity) noexcept {
    reportable_severity = severity;
}

TRTLogger& TRTLogger::getInstance() {
    static TRTLogger instance;
    return instance;
}

std::string TRTLogger::getCurrentTimeStr() const {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%m/%d/%Y-%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity > reportable_severity) {
        return;
    }

    std::string prefix;
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: prefix = "INTERNAL_ERROR"; break;
        case nvinfer1::ILogger::Severity::kERROR:          prefix = "ERROR"; break;
        case nvinfer1::ILogger::Severity::kWARNING:        prefix = "WARNING"; break;
        case nvinfer1::ILogger::Severity::kINFO:           prefix = "INFO"; break;
        case nvinfer1::ILogger::Severity::kVERBOSE:        prefix = "VERBOSE"; break;
        default:                                           prefix = "UNKNOWN"; break;
    }

    std::cerr << "[" << getCurrentTimeStr() << "] [" << prefix << "] " << msg << std::endl;
}
static inline bool file_exists(const std::string& name) {
    struct stat buffer{};
    return (stat(name.c_str(), &buffer) == 0);
}

// helper
 size_t TrtModel::volume(const nvinfer1::Dims& d) 
{ 
    size_t v = 1;
    for (int i=0; i<d.nbDims; i++)
        v *= (d.d[i] > 0 ? static_cast<size_t>(d.d[i]) : 1);  
    return v;   
};


TrtModel::TrtModel(std::string onnxfilepath, bool fp16)
    : onnx_file_path(std::move(onnxfilepath)), FP16(fp16)
{
    const auto idx = onnx_file_path.find(".onnx");
    const auto basename = onnx_file_path.substr(0, idx);
    m_enginePath = basename + ".engine";

    if (file_exists(m_enginePath)){
        std::cout << "start building model from engine file: " << m_enginePath << std::endl;
        this->Runtime();
    } else {
        std::cout << "start building model from onnx file: " << onnx_file_path << std::endl;
        this->genEngine();
        this->Runtime();
    }

    this->trtIOMemory();

}


bool TrtModel::genEngine(){

    TRTLogger& logger = TRTLogger::getInstance();
    logger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);

    // 创建builder
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    if(!builder){
        std::cout << " (T_T)~~~, Failed to create builder."<<std::endl;
        return false;
    }

    // 声明显性batch，创建network
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = make_nvshared(builder->createNetworkV2(explicitBatch));
    if(!network){
        std::cout << " (T_T)~~~, Failed to create network."<<std::endl;
        return false;
    }

    // 创建 config
    auto config = make_nvshared(builder->createBuilderConfig());
    if(!config){
        std::cout << " (T_T)~~~, Failed to create config."<<std::endl;
        return false;
    }

    // 创建parser 从onnx自动构建模型，否则需要自己构建每个算子
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger)); 
    if(!parser){
        std::cout << " (T_T)~~~, Failed to create parser."<<std::endl;
        return false;
    }

    auto parsed = parser->parseFromFile(onnx_file_path.c_str(), 2);
    if(!parsed){
        std::cout << " (T_T)~~~ ,Failed to parse onnx file."<<std::endl;
        return false;
    }

    auto profile = builder->createOptimizationProfile();                                                          

    config->addOptimizationProfile(profile);

    // 判断是否使用半精度优化模型
    if(FP16)  config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // // DLA 仅在可用时开启
    // const int numDLACores = builder->getNbDLACores();
    // if (numDLACores > 0) {
    //     config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    //     config->setDLACore(0);
    //     config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    //     std::cout << "[TRT] Using DLA core 0 with GPU fallback" << std::endl;
    // } else {
    //     config->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);
    // }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28);      /*在新的版本中被使用*/

    // 创建序列化引擎文件
    auto plan = make_nvshared(builder->buildSerializedNetwork(*network, *config));

    if(!plan){
        std::cout << " (T_T)~~~, Failed to SerializedNetwork."<<std::endl;
        return false;
    }

    // 序列化保存推理引擎文件文件
    std::ofstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.good()){
        std::cout << " (T_T)~~~, Failed to open engine file"<<std::endl;
        return false;
    }
    engine_file.write((char *)plan->data(), plan->size());
    engine_file.close();

    std::cout << " ~~Congratulations! 🎉🎉🎉~  Engine build success!!! ✨✨✨~~ " << std::endl;

    return true;

}


std::vector<unsigned char> TrtModel::load_engine_file(){
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(m_enginePath, std::ios::binary);
    if(!engine_file.is_open()) { std::cerr << "[TRT] open engine failed" << std::endl; return engine_data; }
    
    engine_file.seekg(0, std::ios::end);
    const auto length = static_cast<size_t>(engine_file.tellg());
    engine_data.resize(length);
    engine_file.seekg(0, std::ios::beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}


bool TrtModel::Runtime(){
   
    TRTLogger& logger = TRTLogger::getInstance();
    logger.setReportableSeverity(nvinfer1::ILogger::Severity::kINFO); // 或 kVERBOSE

    initLibNvInferPlugins(&logger, "");

    auto plan = load_engine_file();
    if (plan.empty()) { std::cerr << " (T_T)~~~, Failed to load TensorRT engine file." << std::endl; return false; }

    m_runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    if(!m_runtime) { std::cerr << "create runtime failed" << std::endl; return false; }

    m_engine = make_nvshared(m_runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if(!m_engine) { std::cerr << "deserialize failed" << std::endl; return false; }

    const int nbIOTensors = m_engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = m_engine->getIOTensorName(i);
        bool isInput = m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        // auto dims = m_engine->getTensorShape(name);
        // auto dtype = m_engine->getTensorDataType(name);
        // std::cout << "[TRT] Tensor[" <<i<<"] " << name << ", isInput=" << isInput 
        //           << ", dims=" << dims << ", dtype=" << static_cast<int>(dtype) << std::endl;
        if (isInput) {
            m_inputName = name;
        } else {
            m_outputNames.push_back(name);
        }
    }

    if (m_outputNames.empty()) {
        std::cerr << "No output tensors found" << std::endl;
        return false;
    }

    m_context = make_nvshared(m_engine->createExecutionContext());
    if(!m_context) { std::cerr << "create context failed" << std::endl; return false; }

    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

    std::cout << "runtime ready" << std::endl;
    return true;
}


bool TrtModel::trtIOMemory() {
    // Input dims: query using tensor name
    m_inputDims = m_context->getTensorShape(m_inputName.c_str());

    // Input H/W
    if (m_inputDims.nbDims >= 4) {
        // assume dims are N,C,H,W
        config_.resolution = m_inputDims.d[2];
    } else if (m_inputDims.nbDims == 3) {
        // maybe C,H,W
        config_.resolution = m_inputDims.d[2];
    } else {
        std::cerr << "Unsupported input dim layout" << std::endl;
        return false;
    }

    m_imgArea = config_.resolution * config_.resolution;
    m_inputSize = static_cast<int>(TrtModel::volume(m_inputDims) * sizeof(float));

    // Determine which output tensor corresponds to detection and which to protos (segment)
    std::string detectTensorName;
    std::string labelTensorName;
    if (m_outputNames.size() == 1) {
        // only one output -> assume it's detection
        detectTensorName = m_outputNames[0];
    } else {
        // try to distinguish by nbDims number
        auto dims0 = m_context->getTensorShape(m_outputNames[0].c_str());
        auto dims1 = m_context->getTensorShape(m_outputNames[1].c_str());
        
        if (dims0.d[2] == 4) { // protos typically 4-D (N, C, H, W)
            detectTensorName = m_outputNames[0];
            labelTensorName = m_outputNames[1];
        } else if (dims1.d[2] == 4) {
            labelTensorName = m_outputNames[0];
            detectTensorName = m_outputNames[1];
        } else {
            detectTensorName = m_outputNames[0];
            labelTensorName = (m_outputNames.size() > 1 ? m_outputNames[1] : "");
        }
    }

    if (!detectTensorName.empty()) {
        m_detectDims = m_context->getTensorShape(detectTensorName.c_str());
        m_detectSize = static_cast<int>(TrtModel::volume(m_detectDims) * sizeof(float));
    }
    if (!labelTensorName.empty()) {
        m_labelDims = m_context->getTensorShape(labelTensorName.c_str());
        m_labelSize = static_cast<int>(TrtModel::volume(m_labelDims) * sizeof(float));
    }

    // Allocate host/device memory
    CUDA_CHECK(cudaMallocHost(&m_input_host_memory, m_inputSize));
    CUDA_CHECK(cudaMalloc(&m_input_device_memory, m_inputSize));

    if (m_detectSize > 0) CUDA_CHECK(cudaMallocHost(&m_output_host_memory[0], m_detectSize));
    if (m_labelSize > 0) CUDA_CHECK(cudaMallocHost(&m_output_host_memory[1], m_labelSize));

    if (m_detectSize > 0) CUDA_CHECK(cudaMalloc(&m_output_device_memory[0], m_detectSize));
    if (m_labelSize > 0) CUDA_CHECK(cudaMalloc(&m_output_device_memory[1], m_labelSize));

    // Set tensor addresses in context (once, since buffers are fixed)
    m_context->setTensorAddress(m_inputName.c_str(), m_input_device_memory);
    if (!detectTensorName.empty()) m_context->setTensorAddress(detectTensorName.c_str(), m_output_device_memory[0]);
    if (!labelTensorName.empty()) m_context->setTensorAddress(labelTensorName.c_str(), m_output_device_memory[1]);

    // Create CUDA Graph
    m_useGraph = createCudaGraph();
    if (m_useGraph) {
        std::cout << "CUDA Graph created successfully." << std::endl;
    } else {
        std::cout << "Failed to create CUDA Graph, falling back to enqueueV3." << std::endl;
    }

    return true;
}

bool TrtModel::createCudaGraph() {
    cudaGraph_t graph;
    cudaError_t err;

    // Synchronize stream before capture
    err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Begin capture
    err = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamBeginCapture failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Enqueue the inference
    bool status = m_context->enqueueV3(m_stream);
    if (!status) {
        std::cerr << "enqueueV3 failed during graph capture." << std::endl;
        cudaStreamEndCapture(m_stream, nullptr);
        return false;
    }

    // End capture
    err = cudaStreamEndCapture(m_stream, &graph);
    if (err != cudaSuccess) {
        std::cerr << "cudaStreamEndCapture failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Instantiate the graph
    err = cudaGraphInstantiate(&m_graphExec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphInstantiate failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}


void TrtModel::preprocess(const cv::Mat& image) {
    if (image.empty()) throw std::runtime_error("Input image is empty.");
    if (image.channels() != 3) throw std::runtime_error("Input image must be 3-channel BGR.");

    // 1. Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(config_.resolution, config_.resolution), 0, 0, cv::INTER_LINEAR);

    // 2. BGR -> RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3. 转 float32
    resized.convertTo(resized, CV_32FC3);

    // 4. 确保连续内存
    if (!resized.isContinuous()) resized = resized.clone();

    // 5. 指针获取
    const float* src = reinterpret_cast<float*>(resized.data);  // HWC RGB
    float* dst = m_input_host_memory;

    // 6. CHW 三通道指针
    float* dst_r = dst;
    float* dst_g = dst + m_imgArea;
    float* dst_b = dst + 2 * m_imgArea;

    const float inv255 = 1.0f / 255.0f;
    const float mean_r = config_.means[0];
    const float mean_g = config_.means[1];
    const float mean_b = config_.means[2];
    const float std_r  = config_.stds[0];
    const float std_g  = config_.stds[1];
    const float std_b  = config_.stds[2];

    // 7. 高性能指针循环
    const float* p = src;
    for (int i = 0; i < m_imgArea; ++i) {
        float r = (*p++) * inv255;
        float g = (*p++) * inv255;
        float b = (*p++) * inv255;

        dst_r[i] = (r - mean_r) / std_r;
        dst_g[i] = (g - mean_g) / std_g;
        dst_b[i] = (b - mean_b) / std_b;
    }

    // 8. H2D 拷贝
    CUDA_CHECK(cudaMemcpyAsync(
        m_input_device_memory,
        m_input_host_memory,
        m_inputSize,
        cudaMemcpyHostToDevice,
        m_stream
    ));
}


std::vector<detectRes> TrtModel::postprocess(cv::Mat& frame) {
    std::vector<detectRes> results;
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    this->preprocess(frame);

    // 推理
    if (m_useGraph) {
        CUDA_CHECK(cudaGraphLaunch(m_graphExec, m_stream));
    } else {
        if (!m_context->enqueueV3(m_stream)) return results;
    }

    // 拷贝输出
    CUDA_CHECK(cudaMemcpyAsync(m_output_host_memory[0], m_output_device_memory[0], m_detectSize, cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_output_host_memory[1], m_output_device_memory[1], m_labelSize, cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    const float* boxes = m_output_host_memory[0];   // [1,300,4]
    const float* logits = m_output_host_memory[1];  // [1,300,91]

    const int num_query = m_detectDims.d[1];
    const int num_cls = m_labelDims.d[2];

    float scale_w = float(orig_w) / config_.resolution;
    float scale_h = float(orig_h) / config_.resolution;

    // Step1: 收集所有候选 query
    std::vector<TmpDet> tmp; tmp.reserve(num_query);

    for (int i = 0; i < num_query; ++i) {
        int cls_off = i * num_cls;
        float best_score = -1.f;
        int best_cls = -1;

        for (int c = 0; c < num_cls; ++c) {
            float s = 1.f / (1.f + std::exp(-logits[cls_off + c]));  // sigmoid
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        tmp.push_back({i, best_cls, best_score});
    }

    // Step2: TopK
    const int topk = 100;
    std::sort(tmp.begin(), tmp.end(), [](const TmpDet& a, const TmpDet& b){ return a.score > b.score; });
    int keep = std::min(topk, (int)tmp.size());

    // Step3: threshold + decode
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(80, 150);
    for (int k = 0; k < keep; ++k) {
        const auto& t = tmp[k];
        if (t.score < kConfThresh) continue;

        int i = t.q;
        int off = i * 4;
        float cx = boxes[off + 0] * config_.resolution;
        float cy = boxes[off + 1] * config_.resolution;
        float w  = boxes[off + 2] * config_.resolution;
        float h  = boxes[off + 3] * config_.resolution;

        float x1 = (cx - w*0.5f) * scale_w;
        float y1 = (cy - h*0.5f) * scale_h;
        float x2 = (cx + w*0.5f) * scale_w;
        float y2 = (cy + h*0.5f) * scale_h;

        detectRes det;
        det.label = t.cls ;
        // if(det.label < 0) continue;
        det.confidence = t.score;
        det.box = cv::Rect(int(x1), int(y1), int(x2-x1), int(y2-y1));
        // det.box_color = cv::Scalar(0, 255, 0);
        det.box_color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        results.push_back(det);
    }

    return results;
}


bool TrtModel::drawResult(cv::Mat& image, const std::vector<detectRes>& result) {
    int detections = result.size();
    
    for (int i = 0; i < detections; ++i)
    {
        detectRes detection = result[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.box_color;

        // Detection box
        cv::rectangle(image, box, color, 2);

        // Detection box text
        std::string classString = std::to_string(detection.label) + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }

    return true; // 如果需要返回状态，可以返回true
}


TrtModel::~TrtModel()
{
    if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
    if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }

    if (m_input_host_memory) { cudaFreeHost(m_input_host_memory); m_input_host_memory = nullptr; }
    if (m_input_device_memory) { cudaFree(m_input_device_memory); m_input_device_memory = nullptr; }

    for (auto &h : m_output_host_memory) {
        if (h) { cudaFreeHost(h); h = nullptr; }
    }
    for (auto &d : m_output_device_memory) {
        if (d) { cudaFree(d); d = nullptr; }
    }

    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
}
