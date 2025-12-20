# Xavier_Yolov5: YOLOv5 实时目标检测系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English Version](./README_EN.md) | 中文版

## 📋 项目简介
这是一个基于 NVIDIA Xavier 开发板和 DeepStream SDK 的 YOLOv5 实时目标检测系统。该系统实现了高性能、低延迟的目标检测功能，适用于智能监控、自动驾驶辅助等实时计算机视觉应用场景。

## 🎯 主要特性
- **实时目标检测**：基于 YOLOv5 算法，支持多种目标类别的实时检测
- **高性能优化**：针对 NVIDIA Xavier 硬件进行深度优化，实现低延迟推理
- **多线程处理**：采用多线程架构，实现并行 NMS 处理和高效的视频流处理
- **智能目标跟踪**：集成增强的 Kalman 滤波和匈牙利算法，实现稳定的目标跟踪
- **轨迹预测**：基于多项式回归的轨迹预测功能，支持未来位置预测
- **运动模式识别**：能够识别目标的运动模式（静止、线性、曲线、加速、减速、Z字形等）
- **配置化设计**：支持通过配置文件灵活调整检测参数和系统行为
- **动态阈值调整**：根据场景复杂度动态调整检测阈值，提高检测准确性
- **详细日志系统**：实现了分级日志（DEBUG/INFO/WARNING/ERROR/CRITICAL），便于调试和监控
- **高级性能监控**：集成增强的性能分析工具，支持内存使用分析、延迟百分位数统计等
- **可扩展性**：模块化设计，支持自定义网络结构和检测类别
- **自适应噪声处理**：卡尔曼滤波器支持自适应噪声参数调整，提高跟踪稳定性

## 📁 项目结构
```
Xavier_Yolov5/
├── nvdsinfer_custom_impl_yolov5/    # YOLOv5 解析器实现
│   ├── nvdsparsebbox_Yolo.cpp      # 核心解析器代码（已增强）
│   ├── trt_utils.cpp               # TensorRT 工具函数
│   ├── trt_utils.h                 # TensorRT 工具函数头文件
│   ├── Makefile                    # 编译配置
│   └── libnvdsinfer_custom_impl_Yolo.so  # 编译后的共享库
├── config_infer_primary.txt        # 基础推理配置
├── config_infer_primary_V5.txt     # YOLOv5 特定推理配置
├── deepstream_app_config.txt       # 基础 DeepStream 应用配置
├── deepstream_app_config_yoloV5.txt # YOLOv5 特定 DeepStream 应用配置
├── labels.txt                      # 检测类别标签
├── Inkedpic.jpg                    # 项目截图
├── README.md                       # 中文说明文档
└── README_EN.md                    # 英文说明文档
```

## 🛠️ 环境要求
- **硬件**：NVIDIA Xavier NX/Jetson AGX Xavier
- **操作系统**：Ubuntu 18.04 LTS (JetPack 4.5+)
- **依赖库**：
  - NVIDIA JetPack SDK 4.5+
  - DeepStream SDK 5.0+
  - TensorRT 7.1+
  - CUDA 10.2+
  - cuDNN 8.0+
  - OpenCV 4.1+
  - PyTorch 1.9+ (用于模型转换)

## 📦 安装与配置

### 1. 开发板初始化
- 按照 NVIDIA 官方指南刷写 JetPack SDK 到 Xavier 开发板
- 安装必要的依赖库：
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-dev
  pip3 install torch torchvision
  ```

### 2. 模型转换
将训练好的 YOLOv5 模型 (.pth) 转换为 TensorRT 引擎 (.engine)：

```bash
# 克隆 YOLOv5 代码库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 安装依赖
pip3 install -r requirements.txt

# 转换模型（将 best.pt 替换为你的模型文件）
python3 export.py --weights best.pt --img 640 --batch 1 --device 0 --include engine
```

### 3. 项目配置
- 将转换后的 `.engine` 文件和对应的 `labels.txt` 复制到项目根目录
- 根据需要修改配置文件：
  - `deepstream_app_config_yoloV5.txt`：调整视频源、输出设置、显示参数
  - `config_infer_primary_V5.txt`：调整推理参数、置信度阈值等

## 🚀 运行系统

### 1. 编译 YOLOv5 解析器
```bash
cd nvdsinfer_custom_impl_yolov5
make
```

### 2. 启动 DeepStream 应用
```bash
deepstream-app -c ./deepstream_app_config_yoloV5.txt
```

### 3. 查看结果
- **本地显示**：DeepStream 会自动创建显示窗口（如果开发板连接了显示器）
- **远程访问**：使用 RTSP 客户端（如 PotPlayer、VLC）连接到 `rtsp://<Xavier-IP>:8554/ds-test`

## 🔧 核心功能模块

### 1. YOLOv5 解析器 (`nvdsparsebbox_Yolo.cpp`)
- **并行 NMS 处理**：采用多线程架构，对每个检测类别并行执行非极大值抑制
- **高级日志系统**：实现分级日志，便于调试和监控
- **配置管理**：支持通过配置文件动态调整检测参数
- **性能分析**：集成增强的性能分析工具，实时监控系统运行状态
- **错误处理**：完善的异常处理机制，确保系统稳定性

### 2. 智能目标跟踪系统
- **增强的卡尔曼滤波器**：支持 6 状态变量（位置、速度、加速度）的预测和跟踪
- **自适应噪声处理**：根据测量残差动态调整噪声参数，提高跟踪稳定性
- **匈牙利算法匹配**：基于 IoU、速度、大小、类别和距离的多因素成本矩阵进行最优匹配
- **跟踪统计分析**：记录跟踪成功率、类别分布等关键指标

### 3. 轨迹预测与运动分析
- **多项式回归预测**：支持自定义阶数的多项式轨迹拟合和未来位置预测
- **运动模式识别**：能够识别多种运动模式（静止、线性、曲线、加速、减速、Z字形等）
- **轨迹分析**：基于历史轨迹数据进行速度、方向和加速度的统计分析
- **动态阈值调整**：根据场景复杂度动态调整检测阈值，提高检测准确性

### 4. 高级性能分析
- **内存使用监控**：实时跟踪内存占用情况（当前/峰值）
- **详细性能指标**：包括平均延迟、最小/最大延迟、标准偏差和百分位数统计（50%/90%/95%/99%）
- **吞吐量分析**：计算系统处理速率（操作/秒）
- **运行时警告**：对慢操作自动发出警告，便于性能瓶颈识别

### 5. 配置管理系统
- **增强的配置解析**：支持多种数据类型（整数、浮点数、布尔值、字符串、向量等）
- **参数验证**：对配置参数进行范围检查和有效性验证
- **默认值支持**：为可选参数提供合理的默认值
- **配置描述**：为每个配置参数提供详细说明，便于理解和使用

### 6. 性能优化
- **多线程处理**：使用 `std::thread` 和 `std::mutex` 实现线程安全的并行处理
- **内存优化**：采用高效的内存管理策略，减少内存占用
- **算法优化**：优化的 NMS 算法，减少计算复杂度
- **缓存优化**：利用硬件缓存特性，提高数据访问效率

## 📊 性能指标
| 指标 | 值 |
|------|-----|
| 输入分辨率 | 640x640 |
| 推理延迟 | < 20 ms |
| 帧率 | > 30 FPS |
| GPU 占用 | < 50% |
| CPU 占用 | < 20% |

## 🎨 系统架构
```
视频源 → DeepStream → YOLOv5 推理 → 并行 NMS 处理 → 结果可视化/输出
```

## 🔍 关键技术点

### 1. 多线程 NMS 处理
采用 `ParallelNMSProcessor` 类实现并行 NMS 处理：
```cpp
// 创建 NMS 处理器
ParallelNMSProcessor nmsProcessor(maxWorkerThreads);

// 为每个类别提交 NMS 任务
for(int cls_id = 0; cls_id < numClasses; ++cls_id) {
    if (!binfo[cls_id].empty()) {
        nmsProcessor.processClass(cls_id, binfo[cls_id], nmsThreshold, objects, resultMutex);
    }
}

// 等待所有任务完成
nmsProcessor.waitForCompletion();
```

### 2. 配置管理系统
实现了 `ConfigManager` 类，支持从文件加载配置：
```cpp
// 加载配置
ConfigManager configManager;
configManager.loadConfig("config_yoloV5.txt");

// 获取配置参数
int numClasses = configManager.getValue("NUM_CLASSES_YOLO", 4);
float nmsThreshold = configManager.getValue("NMS_THRESHOLD", 0.5f);
```

### 3. 日志系统
实现了分级日志系统，支持不同级别的日志输出：
```cpp
LOG_DEBUG("Debug message");
LOG_INFO("Info message");
LOG_WARNING("Warning message");
LOG_ERROR("Error message");
LOG_CRITICAL("Critical message");
```

## ❗ 注意事项
1. 确保模型文件 (.engine) 与配置文件中的类别数量一致
2. 调整视频源地址时，确保网络连接正常
3. 运行前请检查所有依赖库是否正确安装
4. 对于自定义网络结构，需要相应调整解析器代码

## 🤝 贡献
欢迎提交 Issue 和 Pull Request 来帮助改进这个项目！

## 📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

## 📞 联系方式
如有问题或建议，请通过以下方式联系：
- GitHub: [seamasZ](https://github.com/seamasZ)
- Email: your-email@example.com

---

![image](https://github.com/seamasZ/Xavier_Yolov5/blob/main/Inkedpic.jpg)

*图：系统运行界面展示*
