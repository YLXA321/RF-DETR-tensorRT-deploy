#include <iostream>

#include "TrtModel.hpp"
#include "tracker_object_tools.hpp"


int X {1141}, Y {316}, W {306}, H {344};


// 用来绘制检测目标框
void drawResult(cv::Mat& image, const std::vector<detectRes>& outputs)
{
    int detections = outputs.size();
    for (int i = 0; i < detections; ++i)
    {
        detectRes detection = outputs[i];

        // cv::Rect box = detection.box;
        int x = detection.box.x + X;
        int y = detection.box.y + Y;
        int width = detection.box.width;
        int height = detection.box.height;
        cv::Scalar color = detection.box_color;
        cv::Rect box = cv::Rect(x, y, width, height);

        // Detection box
        cv::rectangle(image, box, color, 2);

        // Detection box text
        std::string classString = std::to_string(detection.label) + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
}

// 用来绘制追踪的目标框
void draw_tracker_bboxes(cv::Mat& frame, const std::vector<byte_track::BYTETracker::STrackPtr>& output)
{
    for (size_t i = 0; i < output.size(); i++) {
        auto detection = output[i];
        auto trackId = detection->getTrackId();

        float x = detection->getRect().tlwh[0]+X;
        float y = detection->getRect().tlwh[1]+Y;
        float width = detection->getRect().tlwh[2];
        float height = detection->getRect().tlwh[3];

        cv::rectangle(frame, cv::Point(x, y), cv::Point(x+width, y+height), cv::Scalar(251, 81, 163), 3);


        // Detection box text
        std::string classString = std::to_string(trackId);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(x, y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, cv::Scalar(125, 40, 81), cv::FILLED);
        cv::putText(frame, classString, cv::Point(x + 5, y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(253, 168, 208), 2, 0);
    }
}


int main() {

    static TrtModel trtmodel("weights/best96.onnx", true);
    byte_track::BYTETracker tracker(30, 3);
    std::vector<byte_track::Object> tracks{};
    TrackObjectTools track_object_tools{};

    cv::VideoCapture cap("media/2.mp4");

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
    int total_count = 0;


    // 读取和显示视频帧，直到视频结束
    while (cap.read(frame)) {

        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Rect roi(X, Y, W, H);
        cv::Mat frame1 = frame(roi).clone();

        auto detectres = trtmodel.detect_postprocess(frame1);

        track_object_tools.format_tracker_input(detectres,tracks); // 转换检测数据结构
        
        const auto outputs = tracker.update(tracks);  // 更新追踪信息,追踪当前所有视频帧的检测目标

        // track_object_tools.draw_tracker_bboxes(frame, outputs);
        draw_tracker_bboxes(frame, outputs);
        // auto aaa = track_object_tools.object_tracker_count(frame_nums, outputs, 120);
        // 获取当前帧新增计数
        total_count= track_object_tools.object_tracker_count(outputs, 120);
         cv::line(frame, cv::Point(120+X, 0),cv::Point(120+X, 800),cv::Scalar(0, 255, 0),2);
        // total_count = track_object_tools.get_total_count();  // 获取累计总数
        drawResult(frame, detectres);



        // track_object_tools.draw_tracker_bboxes(frame, outputs);
        // // auto aaa = track_object_tools.object_tracker_count(frame_nums, outputs, 120);
        // // 获取当前帧新增计数
        // total_count = track_object_tools.object_tracker_count(outputs, 640);
        // cv::line(frame, cv::Point(640, 0),cv::Point(640, 544),cv::Scalar(0, 255, 0),2);
        // // total_count = track_object_tools.get_total_count();  // 获取累计总数
        // // trtmodel.det_drawResult(frame, detectres);


        cv::putText(frame, "duck_nums: " + std::to_string(total_count), cv::Point(10, 100),cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 5);


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


        // // 显示处理后的帧
        // cv::imshow("Processed-trans-Video", frame);
        // 创建一个新图像用于存储缩放后的结果
        cv::Mat resized_frame;
        // 将图像长宽都变为原来的一半
        cv::resize(frame, resized_frame, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        // 显示缩放后的图像
        cv::imshow("Processed-trans-Video", resized_frame);
        frame_nums += 1;
        std::string filename = "./111/" + std::to_string(frame_nums) + ".jpg";
        // cv::imwrite(filename, frame);

        if (cv::waitKey(25) == 27) {
            break;
        }
    }


    // 释放视频捕获对象和关闭所有窗口
    cap.release();
    cv::destroyAllWindows();


    return 0;
}