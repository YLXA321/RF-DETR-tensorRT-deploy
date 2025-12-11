RF-DETR 模型 TensorRT 部署项目说明
本项目基于 RF-DETR 模型，完成了从数据准备、模型训练、ONNX 导出到 TensorRT 推理部署 的完整流程。

📁 项目结构
text
编辑
.
├── python/
│   ├── yolo2coco.py        # 将 YOLO 格式数据集转换为 COCO 格式
│   ├── train_det.py        # RF-DETR 模型训练入口
│   ├── export_det.py       # 将训练好的 .pth 模型导出为 ONNX 格式
│   ├── predict_det.py      # 基于 RF-DETR 的 Python 推理 Demo
│   └── onnx_inference.py   # 使用 ONNX Runtime 进行推理的 Demo
│
├── src/                    # C++ TensorRT 部署核心代码（含前处理、后处理、模型加载等）
├── main.cpp                # C++ 推理调用示例入口
├── weights/                # 存放用于推理的模型文件（如 .onnx 或 .engine）
└── README.md               # 本说明文件
🧾 数据准备
原始格式：YOLO 格式（每张图像对应一个 .txt 标注文件）
目标格式：COCO 格式（RF-DETR 训练所需）
转换脚本：
bash
编辑
python ./python/yolo2coco.py
该脚本将 YOLO 格式的数据集自动转换为标准 COCO JSON 格式，供后续训练使用。
🏋️ 模型训练
使用 train_det.py 启动 RF-DETR 模型训练：

bash
编辑
python ./python/train_det.py
请确保已准备好 COCO 格式的数据集，并在配置中指定路径。

🔁 模型导出（PyTorch → ONNX）
训练完成后，使用以下命令将 .pth 模型导出为 ONNX 格式：

bash
编辑
python ./python/export_det.py
生成的 .onnx 文件将保存在 weights/ 目录中，用于后续 TensorRT 部署。

🚀 推理方式
1. Python 推理 Demo（PyTorch / ONNX）
RF-DETR 原生推理：
bash
编辑
python ./python/predict_det.py
ONNX Runtime 推理：
bash
编辑
python ./python/onnx_inference.py
2. C++ TensorRT 高性能推理
构建步骤：
bash
编辑
cmake -S . -B build
cmake --build build -j$(nproc)
运行推理：
bash
编辑
./build/build
该程序调用 src/ 中的 TensorRT 推理引擎，完成图像检测任务，包含完整的前处理、推理、后处理流程。
