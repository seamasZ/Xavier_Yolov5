# Xavier_Yolov5: YOLOv5 Real-time Object Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文版](./README.md) | English Version

## 📋 Project Overview
This is a real-time object detection system based on NVIDIA Xavier development board and DeepStream SDK. The system implements high-performance, low-latency object detection functionality suitable for real-time computer vision applications such as intelligent surveillance and autonomous driving assistance.

---

![image](https://github.com/seamasZ/Xavier_Yolov5/blob/main/Inkedpic.jpg)

*Figure: System operation interface demonstration*

---

## 🎯 Key Features
- **Real-time Object Detection**: Based on YOLOv5 algorithm, supporting real-time detection of multiple object categories
- **High-performance Optimization**: Deeply optimized for NVIDIA Xavier hardware, achieving low-latency inference
- **Multi-threaded Processing**: Adopts multi-threaded architecture to implement parallel NMS processing and efficient video stream processing
- **Intelligent Object Tracking**: Integrated enhanced Kalman filtering and Hungarian algorithm for stable target tracking
- **Trajectory Prediction**: Polynomial regression-based trajectory prediction for future position forecasting
- **Motion Pattern Recognition**: Capable of recognizing target motion patterns (stationary, linear, curved, accelerating, decelerating, zigzag, etc.)
- **Configuration-based Design**: Supports flexible adjustment of detection parameters and system behavior through configuration files
- **Dynamic Threshold Adjustment**: Dynamically adjusts detection thresholds based on scene complexity for improved accuracy
- **Detailed Logging System**: Implements hierarchical logging (DEBUG/INFO/WARNING/ERROR/CRITICAL) for easy debugging and monitoring
- **Advanced Performance Monitoring**: Enhanced performance analysis tools with memory usage tracking and latency percentile statistics
- **Scalability**: Modular design supporting custom network structures and detection categories
- **Adaptive Noise Processing**: Kalman filter supports adaptive noise parameter adjustment for improved tracking stability

## 📁 Project Structure
```
Xavier_Yolov5/
├── nvdsinfer_custom_impl_yolov5/    # YOLOv5 parser implementation
│   ├── nvdsparsebbox_Yolo.cpp      # Core parser code (enhanced)
│   ├── trt_utils.cpp               # TensorRT utility functions
│   ├── trt_utils.h                 # TensorRT utility function headers
│   ├── Makefile                    # Compilation configuration
│   └── libnvdsinfer_custom_impl_Yolo.so  # Compiled shared library
├── config_infer_primary.txt        # Basic inference configuration
├── config_infer_primary_V5.txt     # YOLOv5 specific inference configuration
├── deepstream_app_config.txt       # Basic DeepStream application configuration
├── deepstream_app_config_yoloV5.txt # YOLOv5 specific DeepStream application configuration
├── labels.txt                      # Detection category labels
├── Inkedpic.jpg                    # Project screenshot
├── README.md                       # Chinese documentation
└── README_EN.md                    # English documentation
```

## 🛠️ Environment Requirements
- **Hardware**: NVIDIA Xavier NX/Jetson AGX Xavier
- **Operating System**: Ubuntu 18.04 LTS (JetPack 4.5+)
- **Dependencies**:
  - NVIDIA JetPack SDK 4.5+
  - DeepStream SDK 5.0+
  - TensorRT 7.1+
  - CUDA 10.2+
  - cuDNN 8.0+
  - OpenCV 4.1+
  - PyTorch 1.9+ (for model conversion)

## 📦 Installation and Configuration

### 1. Development Board Initialization
- Flash JetPack SDK to Xavier development board according to NVIDIA official guide
- Install necessary dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-pip python3-dev
  pip3 install torch torchvision
  ```

### 2. Model Conversion
Convert trained YOLOv5 model (.pth) to TensorRT engine (.engine):

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# Install dependencies
pip3 install -r requirements.txt

# Convert model (replace best.pt with your model file)
python3 export.py --weights best.pt --img 640 --batch 1 --device 0 --include engine
```

### 3. Project Configuration
- Copy the converted `.engine` file and corresponding `labels.txt` to the project root directory
- Modify configuration files as needed:
  - `deepstream_app_config_yoloV5.txt`: Adjust video source, output settings, display parameters
  - `config_infer_primary_V5.txt`: Adjust inference parameters, confidence thresholds, etc.

## 🚀 Running the System

### 1. Compile YOLOv5 Parser
```bash
cd nvdsinfer_custom_impl_yolov5
make
```

### 2. Start DeepStream Application
```bash
deepstream-app -c ./deepstream_app_config_yoloV5.txt
```

### 3. View Results
- **Local Display**: DeepStream will automatically create a display window (if the development board is connected to a monitor)
- **Remote Access**: Use an RTSP client (such as PotPlayer, VLC) to connect to `rtsp://<Xavier-IP>:8554/ds-test`

## 🔧 Core Functional Modules

### 1. YOLOv5 Parser (`nvdsparsebbox_Yolo.cpp`)
- **Parallel NMS Processing**: Adopts multi-threaded architecture to perform non-maximum suppression on each detection category in parallel
- **Advanced Logging System**: Implements hierarchical logging for easy debugging and monitoring
- **Configuration Management**: Supports dynamic adjustment of detection parameters through configuration files
- **Performance Analysis**: Integrates enhanced performance analysis tools to real-time monitor system operation status
- **Error Handling**: Complete exception handling mechanism to ensure system stability

### 2. Intelligent Object Tracking System
- **Enhanced Kalman Filter**: Supports 6-state variable (position, velocity, acceleration) prediction and tracking
- **Adaptive Noise Processing**: Dynamically adjusts noise parameters based on measurement residuals for improved tracking stability
- **Hungarian Algorithm Matching**: Optimal matching using multi-factor cost matrix integrating IoU, velocity, size, class, and distance
- **Tracking Statistics Analysis**: Records key metrics such as tracking success rate and class distribution

### 3. Trajectory Prediction & Motion Analysis
- **Polynomial Regression Prediction**: Supports customizable polynomial trajectory fitting and future position prediction
- **Motion Pattern Recognition**: Capable of recognizing multiple motion patterns (stationary, linear, curved, accelerating, decelerating, zigzag, etc.)
- **Trajectory Analysis**: Statistical analysis of speed, direction, and acceleration based on historical trajectory data
- **Dynamic Threshold Adjustment**: Dynamically adjusts detection thresholds based on scene complexity for improved accuracy

### 4. Advanced Performance Analysis
- **Memory Usage Monitoring**: Real-time tracking of memory usage (current/peak)
- **Detailed Performance Metrics**: Including average latency, min/max latency, standard deviation, and percentile statistics (50%/90%/95%/99%)
- **Throughput Analysis**: Calculates system processing rate (operations per second)
- **Runtime Warnings**: Automatically issues warnings for slow operations to facilitate performance bottleneck identification

### 5. Configuration Management System
- **Enhanced Configuration Parsing**: Supports multiple data types (integers, floats, booleans, strings, vectors, etc.)
- **Parameter Validation**: Performs range checks and validity validation on configuration parameters
- **Default Value Support**: Provides reasonable default values for optional parameters
- **Configuration Descriptions**: Detailed descriptions for each configuration parameter for easy understanding and usage

### 6. Performance Optimization
- **Multi-threaded Processing**: Uses `std::thread` and `std::mutex` to implement thread-safe parallel processing
- **Memory Optimization**: Adopts efficient memory management strategies to reduce memory usage
- **Algorithm Optimization**: Optimized NMS algorithm to reduce computational complexity
- **Cache Optimization**: Utilizes hardware cache characteristics to improve data access efficiency

## 📊 Performance Metrics
| Metric | Value |
|--------|-------|
| Input Resolution | 640x640 |
| Inference Latency | < 20 ms |
| Frame Rate | > 30 FPS |
| GPU Usage | < 50% |
| CPU Usage | < 20% |

## 🎨 System Architecture
```
Video Source → DeepStream → YOLOv5 Inference → Parallel NMS Processing → Result Visualization/Output
```

## 🔍 Key Technical Points

### 1. Multi-threaded NMS Processing
Implements parallel NMS processing using `ParallelNMSProcessor` class:
```cpp
// Create NMS processor
ParallelNMSProcessor nmsProcessor(maxWorkerThreads);

// Submit NMS tasks for each category
for(int cls_id = 0; cls_id < numClasses; ++cls_id) {
    if (!binfo[cls_id].empty()) {
        nmsProcessor.processClass(cls_id, binfo[cls_id], nmsThreshold, objects, resultMutex);
    }
}

// Wait for all tasks to complete
nmsProcessor.waitForCompletion();
```

### 2. Configuration Management System
Implements `ConfigManager` class supporting configuration loading from files:
```cpp
// Load configuration
ConfigManager configManager;
configManager.loadConfig("config_yoloV5.txt");

// Get configuration parameters
int numClasses = configManager.getValue("NUM_CLASSES_YOLO", 4);
float nmsThreshold = configManager.getValue("NMS_THRESHOLD", 0.5f);
```

### 3. Logging System
Implements hierarchical logging system supporting different levels of log output:
```cpp
LOG_DEBUG("Debug message");
LOG_INFO("Info message");
LOG_WARNING("Warning message");
LOG_ERROR("Error message");
LOG_CRITICAL("Critical message");
```

## ❗ Notes
1. Ensure that the number of classes in the model file (.engine) matches that in the configuration file
2. Ensure network connection is normal when adjusting video source address
3. Check that all dependencies are correctly installed before running
4. For custom network structures, corresponding adjustments to the parser code are required

## 🤝 Contributing
Contributions are welcome! Please submit Issues and Pull Requests to help improve this project!

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details

## 📞 Contact
For questions or suggestions, please contact:
- GitHub: [seamasZ](https://github.com/seamasZ)
- Email: your-email@example.com