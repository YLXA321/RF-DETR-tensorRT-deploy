#include <iostream>
#include <opencv2/opencv.hpp>
#include "TrtModel.hpp"

int main() {

    static TrtModel trtmodel("weights/inference_model.sim.onnx", false);

    cv::VideoCapture cap("media/123.mp4");

    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file.\n";
        return -1;
    }

    cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // 获取帧率
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "width: " << frameSize.width << " height: " << frameSize.height << " fps: " << video_fps << std::endl;

    cv::Mat frame;
    int frame_nums = 0;

    // 读取和显示视频帧，直到视频结束
    while (cap.read(frame)) {

        auto start = std::chrono::high_resolution_clock::now();
        
        auto output = trtmodel.postprocess(frame);

        std::cout<<" 11111----------------------------   "<<output.size()<<std::endl;
        trtmodel.drawResult(frame, output);

        // cv::putText(frame, "duck_nums: " + std::to_string(123), cv::Point(10, 100),cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 5);

        // // 获取程序结束时间点
        auto end = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::micro>(end - start).count() / 1000.0; 
        double fps = 1000.0 / duration_ms;
        // 格式化FPS文本
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(2) << fps;
        std::string fps_text = ss.str();
        // 在帧上绘制FPS
        cv::putText(frame, fps_text, cv::Point(200, 200), cv::FONT_HERSHEY_DUPLEX, 3, cv::Scalar(0, 255, 0), 2, 0);

        std::cout<<"--fps-- "<<fps_text<<std::endl;

        // // 显示处理后的帧
        // cv::imshow("Processed-trans-Video", frame);

        frame_nums += 1;
        std::string filename = "./111/" + std::to_string(frame_nums) + ".jpg";
        // cv::imwrite(filename, frame);

        // 2. 显示带结果的图像
        cv::namedWindow("Object Detection", cv::WINDOW_NORMAL);  // 可调整大小
        cv::resizeWindow("Object Detection", 1280, 720);
        cv::imshow("Object Detection", frame);

        // 3. 使用waitKey处理事件，并控制帧率/检查退出
        int key = cv::waitKey(20); // 等待20毫秒
        if (key == 27 || key == 'q' || key == 'Q') { // 如果按下ESC或Q键
            std::cout << "Exit requested by user." << std::endl;
            break; // 退出循环
        }
    }

    // 释放视频捕获对象和关闭所有窗口
    cap.release();
    cv::destroyAllWindows();

    return 0;
}



// #include <iostream>
// #include "TrtModel.hpp"  // 假设包含OpenCV头 <opencv2/opencv.hpp>

// int main() {
//     static TrtModel trtmodel("weights/inference_model.sim.onnx", true);

//     cv::Mat image = cv::imread("media/input-image.jpg");
//     if (image.empty()) {
//         std::cerr << "ERROR: Failed to load image 'media/input-image.jpg'. Check path." << std::endl;
//         return -1;
//     }

//     auto output = trtmodel.postprocess(image);

//     std::cout << "-------------" << output.size() << std::endl;  // 已打印1，说明检测成功

//     trtmodel.drawResult(image, output);  // 绘制结果（假设无GUI调用）

//     // 保存结果图像（无头友好）
//     std::string filename = "./111/" + std::to_string(123) + ".jpg";
//     if (!cv::imwrite(filename, image)) {
//         std::cerr << "ERROR: Failed to save image to " << filename << std::endl;
//         return -1;
//     }
//     std::cout << "Result saved to: " << filename << std::endl;

//     // 修复：注释掉 imshow/destroy（避免GTK错误）
//     // cv::imshow("Processed-trans-Video", image);
//     // cv::waitKey(0);  // 如果有，添加此行暂停显示
//     // cv::destroyAllWindows();

//     return 0;
// }