#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>
#include <sstream>
#include <iomanip>
#include <random>
#include <array>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "nvdsinfer_custom_impl.h"
#include "trt_utils.h"

// Configuration parameters
static const int NUM_CLASSES_YOLO = 4;
static const float NMS_THRESHOLD = 0.5;
static const float CLS_THRESHOLD = 0.5;
static const int MAX_BATCH_SIZE = 8;
static const int MAX_WORKER_THREADS = 4;
static const int MAX_OBJECTS_TRACKED = 100;  // Maximum number of objects to track
static const float TRACKING_IOU_THRESHOLD = 0.5;  // IOU threshold for object tracking
static const int TRACKING_MAX_AGE = 5;  // Maximum number of frames to keep an object without detection
static const bool ENABLE_INT8_QUANTIZATION = true;  // Enable INT8 quantization support
static const bool ENABLE_KALMAN_FILTER = true;  // Enable Kalman filter for tracking
static const float KALMAN_Q = 0.01f;  // Process noise covariance
static const float KALMAN_R = 0.1f;  // Measurement noise covariance
static const float TRACKING_DISTANCE_THRESHOLD = 50.0f;  // Max distance for object association
static const int OBJECT_LIFETIME = 30;  // Maximum lifetime of an object in frames
static const bool ENABLE_OBJECT_PREDICTION = true;  // Enable object position prediction
static const bool ENABLE_CLASS_AGNOSTIC_TRACKING = false;  // Enable class-agnostic tracking
static const bool ENABLE_TRAJECTORY_PREDICTION = true;  // Enable trajectory prediction
static const int TRAJECTORY_BUFFER_SIZE = 20;  // Number of past positions to store for trajectory analysis
static const bool ENABLE_SOFT_NMS = false;  // Enable Soft NMS by default
static const float SOFT_NMS_SIGMA = 0.5f;  // Sigma parameter for Soft NMS
static const float SOFT_NMS_SCORE_THRESHOLD = 0.01f;  // Score threshold for Soft NMS
static const bool ENABLE_MOTION_ANALYSIS = true;  // Enable motion analysis for objects
static const float MOTION_THRESHOLD = 1.0f;  // Minimum velocity to consider object in motion
static const bool ENABLE_HEAT_MAP_GENERATION = false;  // Enable heat map generation
static const bool ENABLE_OBJECT_CATEGORY_FILTERING = false;  // Enable filtering by object category

// Helper function to initialize allowed object categories
static std::vector<int> getAllowedObjectCategories() {
    return {0, 1, 2, 3};  // Allow all categories by default
}

static const std::vector<int>& ALLOWED_OBJECT_CATEGORIES = getAllowedObjectCategories();  // Allow all categories by default

// Advanced logging system
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

static LogLevel currentLogLevel = LogLevel::INFO;
static std::mutex logMutex;

class Logger {
public:
    static void log(LogLevel level, const std::string& message, const std::string& function = "") {
        if (level < currentLogLevel) return;
        
        std::lock_guard<std::mutex> lock(logMutex);
        
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        std::string levelStr;
        switch(level) {
            case LogLevel::DEBUG: levelStr = "DEBUG";
break;
            case LogLevel::INFO: levelStr = "INFO";
break;
            case LogLevel::WARNING: levelStr = "WARNING";
break;
            case LogLevel::ERROR: levelStr = "ERROR";
break;
            case LogLevel::CRITICAL: levelStr = "CRITICAL";
break;
        }
        
        std::cout << "[" << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") 
                  << "." << std::setw(3) << std::setfill('0') << ms.count() << "] "
                  << std::setw(8) << std::left << levelStr << " | ";
        
        if (!function.empty()) {
            std::cout << "[" << function << "] ";
        }
        
        std::cout << message << std::endl;
    }
    
    static void setLogLevel(LogLevel level) {
        currentLogLevel = level;
    }
};

#define LOG_DEBUG(msg) Logger::log(LogLevel::DEBUG, msg, __FUNCTION__)
#define LOG_INFO(msg) Logger::log(LogLevel::INFO, msg, __FUNCTION__)
#define LOG_WARNING(msg) Logger::log(LogLevel::WARNING, msg, __FUNCTION__)
#define LOG_ERROR(msg) Logger::log(LogLevel::ERROR, msg, __FUNCTION__)
#define LOG_CRITICAL(msg) Logger::log(LogLevel::CRITICAL, msg, __FUNCTION__)

// Thread-safe queue for processing tasks
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::move(value));
        condition.notify_one();
    }
    
    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this] { return !queue.empty() || stop; });
        
        if (queue.empty() && stop) return false;
        
        value = std::move(queue.front());
        queue.pop();
        return true;
    }
    
    void shutdown() {
        stop = true;
        condition.notify_all();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
};

// Bounding box processing task
struct BBoxProcessingTask {
    int classId;
    std::vector<NvDsInferParseObjectInfo> bboxes;
    float nmsThreshold;
    std::vector<NvDsInferParseObjectInfo>* result;
    std::mutex* resultMutex;
};

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* This is a sample bounding box parsing function for the sample YoloV3 detector model */
static NvDsInferParseObjectInfo convertBBox(const float& bx, const float& by, const float& bw,
                                     const float& bh, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx;
    float yCenter = by;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;
    
    // Clamp coordinates to network boundaries
    x0 = clamp(x0, 0.0f, static_cast<float>(netW));
    y0 = clamp(y0, 0.0f, static_cast<float>(netH));
    x1 = clamp(x1, 0.0f, static_cast<float>(netW));
    y1 = clamp(y1, 0.0f, static_cast<float>(netH));

    // Ensure valid bounding box dimensions
    float width = clamp(x1 - x0, 1.0f, static_cast<float>(netW));
    float height = clamp(y1 - y0, 1.0f, static_cast<float>(netH));
    
    b.left = x0;
    b.width = width;
    b.top = y0;
    b.height = height;

    LOG_DEBUG("Converted bbox: x=" << b.left << ", y=" << b.top << ", width=" << b.width << ", height=" << b.height);
    return b;
}

static void addBBoxProposal(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    try {
        NvDsInferParseObjectInfo bbi = convertBBox(bx, by, bw, bh, netW, netH);
        
        // Validate bounding box
        if (bbi.width < 1.0f || bbi.height < 1.0f || bbi.left < 0 || bbi.top < 0) {
            LOG_DEBUG("Invalid bbox discarded: width=" << bbi.width << ", height=" << bbi.height);
            return;
        }

        bbi.detectionConfidence = maxProb;
        bbi.classId = maxIndex;
        
        std::lock_guard<std::mutex> lock(logMutex);
        binfo.push_back(bbi);
        
        LOG_DEBUG("Added bbox: class=" << maxIndex << ", confidence=" << maxProb);
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in addBBoxProposal: " << e.what());
    } catch (...) {
        LOG_ERROR("Unknown exception in addBBoxProposal");
    }
}


// Helper function for NMS - 1D overlap calculation
static float overlap1D(float x1min, float x1max, float x2min, float x2max) {
    if (x1min > x2min) {
        std::swap(x1min, x2min);
        std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0.0f : std::min(x1max, x2max) - x2min;
}

// Helper function for NMS - IoU calculation
static float computeIoU(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2) {
    float overlapX = overlap1D(
        bbox1.left, bbox1.left + bbox1.width,
        bbox2.left, bbox2.left + bbox2.width
    );
    
    float overlapY = overlap1D(
        bbox1.top, bbox1.top + bbox1.height,
        bbox2.top, bbox2.top + bbox2.height
    );
    
    float area1 = bbox1.width * bbox1.height;
    float area2 = bbox2.width * bbox2.height;
    float overlap2D = overlapX * overlapY;
    float unionArea = area1 + area2 - overlap2D;
    
    return unionArea <= 0.0f ? 0.0f : overlap2D / unionArea;
}

// Enhanced Non-Maximum Suppression algorithm
static std::vector<NvDsInferParseObjectInfo>
nonMaximumSuppression(const float nmsThresh, std::vector<NvDsInferParseObjectInfo> binfo) {
    if (binfo.empty()) {
        return {};
    }
    
    try {
        // Sort by confidence in descending order
        std::stable_sort(binfo.begin(), binfo.end(),
            [](const NvDsInferParseObjectInfo& b1, const NvDsInferParseObjectInfo& b2) {
                return b1.detectionConfidence > b2.detectionConfidence;
            }
        );
        
        LOG_DEBUG("NMS input: " << binfo.size() << " boxes");
        
        std::vector<NvDsInferParseObjectInfo> keep;
        std::vector<bool> suppressed(binfo.size(), false);
        
        // Advanced NMS with early termination
        for (size_t i = 0; i < binfo.size(); ++i) {
            if (suppressed[i]) continue;
            
            keep.push_back(binfo[i]);
            
            // Compare with all higher confidence boxes that haven't been suppressed
            for (size_t j = i + 1; j < binfo.size(); ++j) {
                if (suppressed[j]) continue;
                
                float iou = computeIoU(binfo[i], binfo[j]);
                
                if (iou > nmsThresh) {
                    suppressed[j] = true;
                }
            }
        }
        
        LOG_DEBUG("NMS output: " << keep.size() << " boxes after filtering");
        return keep;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in nonMaximumSuppression: " << e.what());
        return binfo; // Return original if NMS fails
    } catch (...) {
        LOG_ERROR("Unknown exception in nonMaximumSuppression");
        return binfo; // Return original if NMS fails
    }
}

// Worker thread function for parallel NMS processing
static void nmsWorkerThread(ThreadSafeQueue<BBoxProcessingTask>& taskQueue) {
    BBoxProcessingTask task;
    
    while (taskQueue.pop(task)) {
        try {
            LOG_DEBUG("Worker thread processing class " << task.classId << " with " << task.bboxes.size() << " boxes");
            
            // Perform NMS on this class
            std::vector<NvDsInferParseObjectInfo> result = nonMaximumSuppression(task.nmsThreshold, task.bboxes);
            
            // Safely add results to the output
            if (task.result && task.resultMutex) {
                std::lock_guard<std::mutex> lock(*task.resultMutex);
                task.result->insert(task.result->end(), result.begin(), result.end());
            }
            
            LOG_DEBUG("Worker thread completed class " << task.classId);
            
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in NMS worker thread: " << e.what());
        } catch (...) {
            LOG_ERROR("Unknown exception in NMS worker thread");
        }
    }
}

// Parallel NMS processor
class ParallelNMSProcessor {
private:
    ThreadSafeQueue<BBoxProcessingTask> taskQueue;
    std::vector<std::thread> workerThreads;
    
public:
    ParallelNMSProcessor(size_t numThreads = MAX_WORKER_THREADS) {
        // Create worker threads
        for (size_t i = 0; i < numThreads; ++i) {
            workerThreads.emplace_back(nmsWorkerThread, std::ref(taskQueue));
            LOG_INFO("Created NMS worker thread " << i+1);
        }
    }
    
    ~ParallelNMSProcessor() {
        shutdown();
    }
    
    void shutdown() {
        taskQueue.shutdown();
        
        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        workerThreads.clear();
        LOG_INFO("Parallel NMS processor shutdown");
    }
    
    void processClass(int classId, const std::vector<NvDsInferParseObjectInfo>& bboxes,
                     float nmsThreshold, std::vector<NvDsInferParseObjectInfo>& result,
                     std::mutex& resultMutex) {
        BBoxProcessingTask task;
        task.classId = classId;
        task.bboxes = bboxes;
        task.nmsThreshold = nmsThreshold;
        task.result = &result;
        task.resultMutex = &resultMutex;
        
        taskQueue.push(std::move(task));
    }
    
    void waitForCompletion() {
        while (!taskQueue.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};


// Configuration manager for loading settings from file
class ConfigManager {
private:
    std::unordered_map<std::string, std::string> configMap;
    std::unordered_map<std::string, std::string> configDescriptions;
    std::unordered_map<std::string, bool> requiredParams;
    std::mutex configMutex;
    bool isLoaded;
    bool isValidated;
    
public:
    ConfigManager() : isLoaded(false), isValidated(false) {
        // Register required parameters with descriptions
        registerRequiredParam("NUM_CLASSES_YOLO", "Number of object classes in YOLO model");
        registerRequiredParam("NMS_THRESHOLD", "Non-maximum suppression threshold (0.0-1.0)");
        registerRequiredParam("CONF_THRESH", "Confidence threshold for object detection (0.0-1.0)");
        registerRequiredParam("BATCH_SIZE", "Batch size for inference");
        registerRequiredParam("NETWORK_WIDTH", "Network input width");
        registerRequiredParam("NETWORK_HEIGHT", "Network input height");
    }
    
    bool loadConfig(const std::string& configPath = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app-yoloV5/config_yoloV5.txt") {
        try {
            std::ifstream configFile(configPath);
            if (!configFile.is_open()) {
                LOG_WARNING("Config file not found: " << configPath << ", using default values");
                return false;
            }
            
            std::string line;
            while (std::getline(configFile, line)) {
                // Remove comments and whitespace
                size_t commentPos = line.find('#');
                if (commentPos != std::string::npos) {
                    line = line.substr(0, commentPos);
                }
                
                // Trim whitespace
                line.erase(0, line.find_first_not_of(" \t\n\r"));
                line.erase(line.find_last_not_of(" \t\n\r") + 1);
                
                if (line.empty()) continue;
                
                // Parse key-value pairs
                size_t equalsPos = line.find('=');
                if (equalsPos != std::string::npos) {
                    std::string key = line.substr(0, equalsPos);
                    std::string value = line.substr(equalsPos + 1);
                    
                    // Trim whitespace from key and value
                    key.erase(key.find_last_not_of(" \t") + 1);
                    key.erase(0, key.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    
                    std::lock_guard<std::mutex> lock(configMutex);
                    configMap[key] = value;
                }
            }
            
            isLoaded = true;
            LOG_INFO("Config file loaded successfully: " << configPath);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load config file: " << e.what());
            return false;
        }
    }
    
    void registerRequiredParam(const std::string& key, const std::string& description) {
        std::lock_guard<std::mutex> lock(configMutex);
        requiredParams[key] = true;
        configDescriptions[key] = description;
    }
    
    bool validateConfig() {
        std::lock_guard<std::mutex> lock(configMutex);
        bool allRequiredFound = true;
        
        // Check for missing required parameters
        for (const auto& [key, _] : requiredParams) {
            if (configMap.find(key) == configMap.end()) {
                LOG_WARNING("Missing required parameter: " << key << " - " << configDescriptions[key]);
                allRequiredFound = false;
            } else {
                // Validate parameter ranges if applicable
                if (key == "NMS_THRESHOLD" || key == "CONF_THRESH") {
                    float value = std::stof(configMap[key]);
                    if (value < 0.0f || value > 1.0f) {
                        LOG_WARNING("Parameter " << key << " out of range (0.0-1.0): " << value);
                    }
                } else if (key == "BATCH_SIZE" || key == "NUM_CLASSES_YOLO") {
                    int value = std::stoi(configMap[key]);
                    if (value <= 0) {
                        LOG_WARNING("Parameter " << key << " must be positive: " << value);
                    }
                }
            }
        }
        
        isValidated = allRequiredFound;
        return allRequiredFound;
    }
    
    template<typename T>
    T getValue(const std::string& key, const T& defaultValue) {
        std::lock_guard<std::mutex> lock(configMutex);
        auto it = configMap.find(key);
        if (it != configMap.end()) {
            try {
                if constexpr (std::is_same_v<T, int>) {
                    return std::stoi(it->second);
                } else if constexpr (std::is_same_v<T, float>) {
                    return std::stof(it->second);
                } else if constexpr (std::is_same_v<T, double>) {
                    return std::stod(it->second);
                } else if constexpr (std::is_same_v<T, bool>) {
                    std::string val = it->second;
                    std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                    return val == "true" || val == "1";
                } else if constexpr (std::is_same_v<T, std::string>) {
                    return it->second;
                } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                    std::vector<int> result;
                    std::stringstream ss(it->second);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        result.push_back(std::stoi(token));
                    }
                    return result;
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    std::vector<float> result;
                    std::stringstream ss(it->second);
                    std::string token;
                    while (std::getline(ss, token, ',')) {
                        result.push_back(std::stof(token));
                    }
                    return result;
                }
            } catch (...) {
                LOG_WARNING("Failed to parse config value for key: " << key << ", using default");
            }
        }
        return defaultValue;
    }
    
    // Dump all configuration parameters
    void dumpConfig() {
        std::lock_guard<std::mutex> lock(configMutex);
        LOG_INFO("=== Configuration Parameters ===");
        
        for (const auto& [key, value] : configMap) {
            std::string status = (requiredParams.find(key) != requiredParams.end()) ? "[Required]" : "[Optional]";
            std::string desc = (configDescriptions.find(key) != configDescriptions.end()) ? ": " + configDescriptions[key] : "";
            LOG_INFO(status << " " << key << " = " << value << desc);
        }
        
        LOG_INFO("============================");
    }
    
    // Check if config is loaded and validated
    bool isConfigValidated() const {
        return isLoaded && isValidated;
    }
};

// Global config manager instance
static ConfigManager g_configManager;

// Dynamic threshold adjuster based on detection statistics
class DynamicThresholdAdjuster {
private:
    float baseConfidenceThreshold;
    float baseNMSThreshold;
    std::mutex thresholdMutex;
    float currentConfidenceThreshold;
    float currentNMSThreshold;
    int adaptationWindowSize;
    
public:
    DynamicThresholdAdjuster(float confThreshold = 0.5f, float nmsThreshold = 0.5f) 
        : baseConfidenceThreshold(confThreshold), baseNMSThreshold(nmsThreshold), 
          currentConfidenceThreshold(confThreshold), currentNMSThreshold(nmsThreshold),
          adaptationWindowSize(50) {}
    
    // Adjust thresholds based on detection density and quality
    void adjustThresholds(const DetectionStatistics& stats) {
        std::lock_guard<std::mutex> lock(thresholdMutex);
        
        // Get current detection statistics
        auto trackedObjects = g_objectTracker.getTrackedObjects();
        int activeObjects = static_cast<int>(trackedObjects.size());
        
        // Dynamic threshold adjustment logic
        if (activeObjects > 50) {
            // High object density - increase thresholds
            currentConfidenceThreshold = std::min(baseConfidenceThreshold * 1.5f, 0.8f);
            currentNMSThreshold = std::min(baseNMSThreshold * 1.3f, 0.7f);
        } else if (activeObjects < 5) {
            // Low object density - decrease thresholds
            currentConfidenceThreshold = std::max(baseConfidenceThreshold * 0.7f, 0.2f);
            currentNMSThreshold = std::max(baseNMSThreshold * 0.8f, 0.3f);
        } else {
            // Normal object density - use base thresholds
            currentConfidenceThreshold = baseConfidenceThreshold;
            currentNMSThreshold = baseNMSThreshold;
        }
        
        LOG_DEBUG("Dynamic threshold adjustment: Confidence=" << currentConfidenceThreshold 
                 << ", NMS=" << currentNMSThreshold);
    }
    
    float getCurrentConfidenceThreshold() const {
        std::lock_guard<std::mutex> lock(thresholdMutex);
        return currentConfidenceThreshold;
    }
    
    float getCurrentNMSThreshold() const {
        std::lock_guard<std::mutex> lock(thresholdMutex);
        return currentNMSThreshold;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(thresholdMutex);
        currentConfidenceThreshold = baseConfidenceThreshold;
        currentNMSThreshold = baseNMSThreshold;
    }
};

// Performance profiler for measuring execution times
class PerformanceProfiler {
private:
    static const int DEFAULT_WARMUP_FRAMES = 3;  // Number of initial frames to skip for warmup

    struct MeasurementData {
        std::vector<double> times;          // Execution times in ms
        std::vector<double> warmupTimes;    // Warmup execution times (excluded from stats)
        size_t peakMemoryUsage;             // Peak memory usage during measurements
        size_t currentMemoryUsage;          // Current memory usage
        double totalTime;                   // Total time spent (excluding warmup)
        double lastMeasurement;             // Last measurement value
        int measurementCount;               // Number of measurements (excluding warmup)
        int warmupCount;                    // Number of warmup measurements collected
    };
    
    std::unordered_map<std::string, MeasurementData> measurements;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimeMap;
    std::unordered_map<std::string, size_t> startMemoryMap;
    std::mutex profilerMutex;
    bool isEnabled;
    bool enableDetailedProfiling;          // Enable detailed profiling with memory usage
    bool enableMemoryTracking;             // Enable memory tracking
    bool enableCPUTracking;                // Enable CPU usage tracking (if supported)
    size_t globalPeakMemory;               // Global peak memory across all measurements
    std::unordered_map<std::string, double> warningThresholds;  // Warning thresholds for slow operations
    
    // Helper to get current memory usage in bytes
    size_t getCurrentMemoryUsage() {
        // Simple memory usage estimation - in real implementation, this would use platform-specific APIs
        size_t totalMemory = 0;
        
        // Count memory used by measurements
        for (const auto& [name, data] : measurements) {
            totalMemory += data.times.size() * sizeof(double);
            totalMemory += sizeof(MeasurementData);
        }
        
        // Add memory used by maps
        totalMemory += startTimeMap.size() * (sizeof(std::string) + sizeof(std::chrono::high_resolution_clock::time_point));
        totalMemory += startMemoryMap.size() * (sizeof(std::string) + sizeof(size_t));
        totalMemory += warningThresholds.size() * (sizeof(std::string) + sizeof(double));
        
        return totalMemory;
    }
    
    // Helper to calculate percentile values
    double calculatePercentile(const std::vector<double>& data, double percentile) {
        if (data.empty()) return 0.0;
        
        std::vector<double> sortedData = data;
        std::sort(sortedData.begin(), sortedData.end());
        
        size_t index = static_cast<size_t>(std::round(percentile / 100.0 * (sortedData.size() - 1)));
        return sortedData[index];
    }
    
    // Helper to calculate frequency statistics
    double calculateFrequency(const std::vector<double>& data) {
        if (data.size() < 2) return 0.0;
        
        // Calculate average time between measurements
        double totalDiff = 0.0;
        for (size_t i = 1; i < data.size(); i++) {
            totalDiff += data[i] - data[i-1];
        }
        
        double avgDiff = totalDiff / (data.size() - 1);
        return avgDiff > 0.0 ? 1000.0 / avgDiff : 0.0; // Hz
    }
    
public:
    PerformanceProfiler() 
        : isEnabled(true), enableDetailedProfiling(false), enableMemoryTracking(false), 
          enableCPUTracking(false), globalPeakMemory(0) {
        // Set default warning thresholds (ms)
        warningThresholds["ObjectTracker::trackObjects"] = 10.0;
        warningThresholds["Yolo::parseBoundingBox"] = 5.0;
        warningThresholds["ParallelNMSProcessor::process"] = 3.0;
        warningThresholds["BBoxPostProcessor::applySoftNMS"] = 2.0;
    }
    
    void enable() { isEnabled = true; }
    void disable() { isEnabled = false; }
    void enableDetailedProfiling() { enableDetailedProfiling = true; }
    void disableDetailedProfiling() { enableDetailedProfiling = false; }
    
    void startMeasurement(const std::string& name) {
        if (!isEnabled) return;
        
        std::lock_guard<std::mutex> lock(profilerMutex);
        startTimeMap[name] = std::chrono::high_resolution_clock::now();
        
        // Record initial memory usage if memory tracking is enabled
        if (enableMemoryTracking || enableDetailedProfiling) {
            startMemoryMap[name] = getCurrentMemoryUsage();
        }
    }
    
    void endMeasurement(const std::string& name) {
        if (!isEnabled) return;
        
        std::lock_guard<std::mutex> lock(profilerMutex);
        auto it = startTimeMap.find(name);
        if (it != startTimeMap.end()) {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - it->second).count() / 1000.0;
            
            // Update measurement data
            MeasurementData& data = measurements[name];
            data.lastMeasurement = duration;
            
            // Skip warmup frames to get more accurate average inference time
            if (data.warmupCount < DEFAULT_WARMUP_FRAMES) {
                data.warmupTimes.push_back(duration);
                data.warmupCount++;
                LOG_DEBUG("Warmup frame " << data.warmupCount << "/" << DEFAULT_WARMUP_FRAMES
                          << " for " << name << ": " << std::fixed << std::setprecision(3) << duration << "ms (excluded from stats)");
            } else {
                data.times.push_back(duration);
                data.totalTime += duration;
                data.measurementCount++;
            }
            
            // Update memory usage if enabled
            if (enableMemoryTracking || enableDetailedProfiling) {
                auto memIt = startMemoryMap.find(name);
                if (memIt != startMemoryMap.end()) {
                    size_t endMemory = getCurrentMemoryUsage();
                    data.currentMemoryUsage = endMemory;
                    
                    // Update peak memory usage for this measurement
                    if (endMemory > data.peakMemoryUsage) {
                        data.peakMemoryUsage = endMemory;
                    }
                    
                    // Update global peak memory
                    if (endMemory > globalPeakMemory) {
                        globalPeakMemory = endMemory;
                    }
                    
                    startMemoryMap.erase(memIt);
                }
            }
            
            // Check for slow operations and log warnings
            auto thresholdIt = warningThresholds.find(name);
            if (thresholdIt != warningThresholds.end() && duration > thresholdIt->second) {
                LOG_WARNING("Slow operation detected: " << name << " took " << std::fixed << std::setprecision(3) << duration << "ms (threshold: " << thresholdIt->second << "ms)");
            }
            
            startTimeMap.erase(it);
        }
    }
    
    void printStatistics() {
        if (!isEnabled || measurements.empty()) return;
        
        std::lock_guard<std::mutex> lock(profilerMutex);
        LOG_INFO("=== Performance Statistics ===");
        
        // Print overall profiling statistics
        size_t totalMeasurements = 0;
        double totalExecutionTime = 0.0;
        
        for (const auto& [name, data] : measurements) {
            totalMeasurements += data.measurementCount;
            totalExecutionTime += data.totalTime;
        }
        
        LOG_INFO("Total measurements: " << totalMeasurements);
        LOG_INFO("Total execution time: " << std::fixed << std::setprecision(3) << totalExecutionTime << "ms");
        
        // Print memory usage statistics
        if (enableMemoryTracking || enableDetailedProfiling) {
            LOG_INFO("Global peak memory usage: " << globalPeakMemory << " bytes");
            LOG_INFO("Profiler current memory usage: " << getCurrentMemoryUsage() << " bytes");
        }
        
        // Print detailed statistics for each measurement
        for (const auto& [name, data] : measurements) {
            // Print warmup statistics
            if (!data.warmupTimes.empty()) {
                double warmupAvg = 0.0;
                for (double t : data.warmupTimes) warmupAvg += t;
                warmupAvg /= data.warmupTimes.size();
                double warmupMax = *std::max_element(data.warmupTimes.begin(), data.warmupTimes.end());
                LOG_INFO("\n" << name << " Warmup Statistics (" << data.warmupCount << " frames, excluded from avg):");
                LOG_INFO("  Warmup Average: " << std::fixed << std::setprecision(3) << warmupAvg << "ms");
                LOG_INFO("  Warmup Max: " << std::fixed << std::setprecision(3) << warmupMax << "ms");
            }

            if (data.times.empty()) {
                LOG_INFO("\n" << name << ": No valid measurements after warmup (" << data.warmupCount << " warmup frames collected)");
                continue;
            }
            
            // Calculate basic statistics (excluding warmup)
            double avg = data.totalTime / data.measurementCount;
            double min = *std::min_element(data.times.begin(), data.times.end());
            double max = *std::max_element(data.times.begin(), data.times.end());
            
            // Calculate standard deviation
            double variance = 0.0;
            for (double time : data.times) {
                variance += (time - avg) * (time - avg);
            }
            variance /= data.measurementCount;
            double stdDev = std::sqrt(variance);
            
            // Calculate percentiles
            double p50 = calculatePercentile(data.times, 50.0);
            double p90 = calculatePercentile(data.times, 90.0);
            double p95 = calculatePercentile(data.times, 95.0);
            double p99 = calculatePercentile(data.times, 99.0);
            
            // Calculate throughput
            double throughput = data.measurementCount / (data.totalTime / 1000.0); // operations per second
            
            // Start logging statistics for this measurement
            LOG_INFO("\n" << name << " Performance (after " << data.warmupCount << " warmup frames):");
            LOG_INFO("  Inference Time (ms):");
            LOG_INFO("    Average: " << std::fixed << std::setprecision(3) << avg);
            LOG_INFO("    Minimum: " << std::fixed << std::setprecision(3) << min);
            LOG_INFO("    Maximum: " << std::fixed << std::setprecision(3) << max);
            LOG_INFO("    Standard Deviation: " << std::fixed << std::setprecision(3) << stdDev);
            LOG_INFO("    Last Measurement: " << std::fixed << std::setprecision(3) << data.lastMeasurement);
            LOG_INFO("  Percentiles (ms):");
            LOG_INFO("    50th (P50): " << std::fixed << std::setprecision(3) << p50);
            LOG_INFO("    90th (P90): " << std::fixed << std::setprecision(3) << p90);
            LOG_INFO("    95th (P95): " << std::fixed << std::setprecision(3) << p95);
            LOG_INFO("    99th (P99): " << std::fixed << std::setprecision(3) << p99);
            LOG_INFO("  Performance Metrics:");
            LOG_INFO("    Throughput: " << std::fixed << std::setprecision(3) << throughput << " ops/sec");
            LOG_INFO("    Total Inference Time: " << std::fixed << std::setprecision(3) << data.totalTime << "ms");
            LOG_INFO("    Valid Sample Count: " << data.measurementCount << " (excluded " << data.warmupCount << " warmup frames)");
            
            // Print memory usage if enabled
            if (enableMemoryTracking || enableDetailedProfiling) {
                LOG_INFO("  Memory Usage (bytes):");
                LOG_INFO("    Current: " << data.currentMemoryUsage);
                LOG_INFO("    Peak: " << data.peakMemoryUsage);
            }
            
            // Print warning threshold if set
            auto thresholdIt = warningThresholds.find(name);
            if (thresholdIt != warningThresholds.end()) {
                LOG_INFO("  Warning Threshold: " << thresholdIt->second << "ms");
            }
        }
        
        LOG_INFO("\n============================");
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(profilerMutex);
        measurements.clear();
        startTimeMap.clear();
        startMemoryMap.clear();
        globalPeakMemory = 0;
    }
};

// Global dynamic threshold adjuster instance
static DynamicThresholdAdjuster g_thresholdAdjuster;

// Global performance profiler instance
static PerformanceProfiler g_profiler;

// Motion pattern types
enum class MotionPattern {
    STATIONARY,    // Object is not moving
    LINEAR,        // Object is moving in a straight line
    CURVED,        // Object is moving with a curved trajectory
    ACCELERATING,  // Object is accelerating
    DECELERATING,  // Object is decelerating
    ZIGZAG,        // Object is moving in a zigzag pattern
    UNKNOWN        // Motion pattern cannot be determined
};

// Tracked object structure
struct TrackedObject {
    int id;  // Unique tracking ID
    NvDsInferParseObjectInfo bbox;  // Current bounding box
    cv::Point2f centroid;  // Object centroid
    int age;  // Number of consecutive frames tracked
    int timeSinceUpdate;  // Number of frames since last detection
    int lifetime;  // Total frames the object has been tracked
    bool active;  // Whether the object is currently active
    float velocityX;  // X velocity component
    float velocityY;  // Y velocity component
    float accelerationX;  // X acceleration component
    float accelerationY;  // Y acceleration component
    KalmanFilter kf;  // Kalman filter for prediction and smoothing
    std::deque<cv::Point2f> trajectory;  // Trajectory buffer for past positions
    bool isMoving;  // Whether the object is currently moving
    float totalDistance;  // Total distance traveled by the object
    cv::Point2f previousPosition;  // Position in the previous frame
    cv::Point2f previousVelocity;  // Velocity in the previous frame
    MotionPattern motionPattern;  // Current motion pattern
    float speedVariance;  // Variance of speed over time
    float directionVariance;  // Variance of direction over time
    std::deque<float> speedHistory;  // History of speed values
    std::deque<float> directionHistory;  // History of direction angles
    int motionPatternUpdateCounter;  // Counter for motion pattern updates
    
    TrackedObject(int objId, const NvDsInferParseObjectInfo& boundingBox) 
        : id(objId), bbox(boundingBox), age(1), timeSinceUpdate(0), lifetime(1), 
          active(true), velocityX(0.0f), velocityY(0.0f), accelerationX(0.0f), 
          accelerationY(0.0f), kf(), trajectory(), isMoving(false), totalDistance(0.0f),
          motionPattern(MotionPattern::UNKNOWN), speedVariance(0.0f), directionVariance(0.0f),
          speedHistory(), directionHistory(), motionPatternUpdateCounter(0) {
        float cx = bbox.left + bbox.width / 2;
        float cy = bbox.top + bbox.height / 2;
        centroid = cv::Point2f(cx, cy);
        previousPosition = centroid;
        previousVelocity = cv::Point2f(0.0f, 0.0f);
        
        // Initialize Kalman filter with first detection
        kf.initialize(centroid);
        
        // Add initial position to trajectory
        addToTrajectory(centroid);
    }
    
    void update(const NvDsInferParseObjectInfo& boundingBox) {
        // Calculate previous position for velocity/acceleration
        previousPosition = centroid;
        previousVelocity = cv::Point2f(velocityX, velocityY);
        
        // Update Kalman filter with new measurement
        float cx = boundingBox.left + boundingBox.width / 2;
        float cy = boundingBox.top + boundingBox.height / 2;
        cv::Point2f newCentroid(cx, cy);
        
        // Apply Kalman filter to smooth the centroid
        centroid = kf.update(newCentroid);
        
        // Calculate velocity based on Kalman filter state
        cv::Point2f velocity = kf.getCurrentVelocity();
        velocityX = velocity.x;
        velocityY = velocity.y;
        
        // Calculate acceleration
        accelerationX = velocityX - previousVelocity.x;
        accelerationY = velocityY - previousVelocity.y;
        
        // Update total distance traveled
        float distance = cv::norm(centroid - previousPosition);
        totalDistance += distance;
        
        // Update movement state
        float speed = cv::norm(cv::Point2f(velocityX, velocityY));
        isMoving = (speed > MOTION_THRESHOLD);
        
        // Update bounding box
        bbox = boundingBox;
        
        // Update tracking statistics
        age++;
        timeSinceUpdate = 0;
        lifetime++;
        active = true;
        
        // Add current position to trajectory
        addToTrajectory(centroid);
        
        // Update motion pattern based on trajectory analysis
        updateMotionPattern();
    }
    
    void predict() {
        if (ENABLE_OBJECT_PREDICTION && active) {
            // Predict next position using Kalman filter
            cv::Point2f predictedCentroid = kf.predict();
            
            // Calculate the offset from current centroid
            float offsetX = predictedCentroid.x - centroid.x;
            float offsetY = predictedCentroid.y - centroid.y;
            
            // Update bounding box with predicted position
            bbox.left += offsetX;
            bbox.top += offsetY;
            
            // Update centroid
            centroid = predictedCentroid;
            
            // Add predicted position to trajectory if trajectory prediction is enabled
            if (ENABLE_TRAJECTORY_PREDICTION) {
                addToTrajectory(centroid);
            }
        }
    }
    
    void incrementAge() {
        timeSinceUpdate++;
        
        if (timeSinceUpdate > TRACKING_MAX_AGE) {
            active = false;
        } else if (active) {
            // Predict next position even when no detection is available
            predict();
        }
        
        // Also check lifetime
        if (lifetime > OBJECT_LIFETIME) {
            active = false;
        }
    }
    
    // Check if a new detection matches this tracked object
    bool matchesDetection(const NvDsInferParseObjectInfo& detection, float maxDistance) {
        float cx = detection.left + detection.width / 2;
        float cy = detection.top + detection.height / 2;
        cv::Point2f detectionCentroid(cx, cy);
        
        float distance = cv::norm(centroid - detectionCentroid);
        
        // If class-agnostic tracking is disabled, check class ID match
        if (!ENABLE_CLASS_AGNOSTIC_TRACKING && bbox.classId != detection.classId) {
            return false;
        }
        
        // If object category filtering is enabled, check if this category is allowed
        if (ENABLE_OBJECT_CATEGORY_FILTERING && 
            std::find(ALLOWED_OBJECT_CATEGORIES.begin(), ALLOWED_OBJECT_CATEGORIES.end(), 
                     detection.classId) == ALLOWED_OBJECT_CATEGORIES.end()) {
            return false;
        }
        
        // Check if distance is within threshold
        return distance < maxDistance;
    }
    
    // Add position to trajectory buffer
    void addToTrajectory(const cv::Point2f& position) {
        trajectory.push_back(position);
        
        // Maintain trajectory buffer size
        if (trajectory.size() > TRAJECTORY_BUFFER_SIZE) {
            trajectory.pop_front();
        }
    }
    
    // Get the trajectory prediction for the next few frames
    std::vector<cv::Point2f> predictTrajectory(int steps = 5) {
        std::vector<cv::Point2f> prediction;
        
        if (!active || trajectory.empty()) {
            return prediction;
        }
        
        // Simple linear prediction based on current velocity
        cv::Point2f currentVelocity(velocityX, velocityY);
        
        for (int i = 1; i <= steps; ++i) {
            cv::Point2f predicted = centroid + (currentVelocity * static_cast<float>(i));
            prediction.push_back(predicted);
        }
        
        return prediction;
    }
    
    // Get the current speed of the object
    float getSpeed() const {
        return cv::norm(cv::Point2f(velocityX, velocityY));
    }
    
    // Get the average speed over the tracked lifetime
    float getAverageSpeed() const {
        if (lifetime <= 1) {
            return 0.0f;
        }
        return totalDistance / static_cast<float>(lifetime - 1);
    }
    
    // Check if the object is stationary (has been in the same position for a while)
    bool isStationary() const {
        return !isMoving;
    }
    
    // Update motion pattern based on trajectory analysis
    void updateMotionPattern() {
        if (trajectory.size() < 5) {
            motionPattern = MotionPattern::UNKNOWN;
            return;
        }
        
        // Update speed and direction history
        float currentSpeed = getSpeed();
        speedHistory.push_back(currentSpeed);
        if (speedHistory.size() > 10) {
            speedHistory.pop_front();
        }
        
        float currentDirection = std::atan2(velocityY, velocityX);
        directionHistory.push_back(currentDirection);
        if (directionHistory.size() > 10) {
            directionHistory.pop_front();
        }
        
        // Calculate speed variance
        if (speedHistory.size() > 1) {
            float avgSpeed = std::accumulate(speedHistory.begin(), speedHistory.end(), 0.0f) / speedHistory.size();
            float variance = 0.0f;
            for (float speed : speedHistory) {
                variance += std::pow(speed - avgSpeed, 2);
            }
            speedVariance = variance / (speedHistory.size() - 1);
        }
        
        // Calculate direction variance
        if (directionHistory.size() > 1) {
            float avgDirection = std::accumulate(directionHistory.begin(), directionHistory.end(), 0.0f) / directionHistory.size();
            float variance = 0.0f;
            for (float direction : directionHistory) {
                variance += std::pow(direction - avgDirection, 2);
            }
            directionVariance = variance / (directionHistory.size() - 1);
        }
        
        // Determine motion pattern
        if (currentSpeed < 0.5f) {
            motionPattern = MotionPattern::STATIONARY;
        } else if (std::abs(accelerationX) > 0.1f || std::abs(accelerationY) > 0.1f) {
            if ((accelerationX > 0 && velocityX > 0) || (accelerationX < 0 && velocityX < 0) ||
                (accelerationY > 0 && velocityY > 0) || (accelerationY < 0 && velocityY < 0)) {
                motionPattern = MotionPattern::ACCELERATING;
            } else {
                motionPattern = MotionPattern::DECELERATING;
            }
        } else if (directionVariance > 0.1f) {
            if (directionVariance > 0.3f) {
                motionPattern = MotionPattern::ZIGZAG;
            } else {
                motionPattern = MotionPattern::CURVED;
            }
        } else {
            motionPattern = MotionPattern::LINEAR;
        }
        
        motionPatternUpdateCounter++;
    }
    
    // Get current motion pattern
    MotionPattern getMotionPattern() const {
        return motionPattern;
    }
    
    // Polynomial regression for trajectory prediction
    // degree: polynomial degree (1 for linear, 2 for quadratic, etc.)
    // steps: number of future steps to predict
    std::vector<cv::Point2f> predictTrajectoryPolynomial(int steps = 5, int degree = 2) {
        std::vector<cv::Point2f> prediction;
        
        if (trajectory.size() < degree + 1) {
            // Not enough points for polynomial regression, use simple prediction
            return predictTrajectory(steps);
        }
        
        // Extract x and y coordinates from trajectory
        std::vector<float> xData, yData, timeData;
        for (size_t i = 0; i < trajectory.size(); i++) {
            xData.push_back(trajectory[i].x);
            yData.push_back(trajectory[i].y);
            timeData.push_back(static_cast<float>(i));
        }
        
        // Perform polynomial regression for x and y coordinates
        cv::Mat xCoeffs, yCoeffs;
        cv::polyfit(timeData, xData, degree, xCoeffs);
        cv::polyfit(timeData, yData, degree, yCoeffs);
        
        // Predict future positions
        float currentTime = static_cast<float>(trajectory.size() - 1);
        for (int i = 1; i <= steps; i++) {
            float predictTime = currentTime + static_cast<float>(i);
            float predictX = cv::polyval(xCoeffs, predictTime);
            float predictY = cv::polyval(yCoeffs, predictTime);
            prediction.push_back(cv::Point2f(predictX, predictY));
        }
        
        return prediction;
    }
    
    // Get motion pattern as string for logging
    std::string getMotionPatternString() const {
        switch (motionPattern) {
            case MotionPattern::STATIONARY:
                return "STATIONARY";
            case MotionPattern::LINEAR:
                return "LINEAR";
            case MotionPattern::CURVED:
                return "CURVED";
            case MotionPattern::ACCELERATING:
                return "ACCELERATING";
            case MotionPattern::DECELERATING:
                return "DECELERATING";
            case MotionPattern::ZIGZAG:
                return "ZIGZAG";
            default:
                return "UNKNOWN";
        }
    }
};

// Helper function for polynomial regression
template<typename T>
void polyfit(const std::vector<T>& x, const std::vector<T>& y, int degree, cv::Mat& coeffs) {
    int n = static_cast<int>(x.size());
    assert(x.size() == y.size());
    assert(n > degree);
    
    // Create the Vandermonde matrix
    cv::Mat A(n, degree + 1, CV_64F);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= degree; j++) {
            A.at<double>(i, j) = std::pow(x[i], j);
        }
    }
    
    // Create the y vector
    cv::Mat Y(n, 1, CV_64F);
    for (int i = 0; i < n; i++) {
        Y.at<double>(i, 0) = y[i];
    }
    
    // Solve the linear system A*coeffs = Y
    cv::solve(A, Y, coeffs, cv::DECOMP_SVD);
}

// Enhanced object tracker with Kalman filter support
class ObjectTracker {
private:
    std::vector<TrackedObject> trackedObjects;
    int nextId;
    std::mutex trackerMutex;
    
    // Tracking statistics structure
    struct TrackingStats {
        int totalObjectsTracked;
        int objectsCreated;
        int objectsLost;
        int successfulAssignments;
        int failedAssignments;
        double averageTrackLifetime;
        double averageAssignmentCost;
        std::vector<double> assignmentCostHistory;
        std::unordered_map<int, int> classDistribution;
        std::chrono::time_point<std::chrono::system_clock> startTime;
        
        TrackingStats() : totalObjectsTracked(0), objectsCreated(0), objectsLost(0),
                         successfulAssignments(0), failedAssignments(0),
                         averageTrackLifetime(0.0), averageAssignmentCost(0.0) {
            startTime = std::chrono::system_clock::now();
        }
    };
    
    TrackingStats trackingStats;
    
    float calculateCentroidDistance(const cv::Point2f& c1, const cv::Point2f& c2) {
        return cv::norm(c1 - c2);
    }
    
    // Calculate velocity similarity between track and detection
    float calculateVelocitySimilarity(const TrackedObject& track, const NvDsInferParseObjectInfo& detection) {
        // Get track's current velocity
        cv::Point2f trackVelocity(track.velocityX, track.velocityY);
        
        // Calculate detection centroid
        float detectionCx = detection.left + detection.width / 2;
        float detectionCy = detection.top + detection.height / 2;
        cv::Point2f detectionCentroid(detectionCx, detectionCy);
        
        // Calculate expected centroid based on track's velocity
        cv::Point2f expectedCentroid = track.centroid + trackVelocity;
        
        // Calculate velocity similarity as inverse of distance between expected and actual detection
        float distance = cv::norm(expectedCentroid - detectionCentroid);
        return 1.0f / (1.0f + distance); // Normalize to [0, 1]
    }
    
    // Calculate size difference between two bounding boxes
    float calculateSizeDifference(const NvDsInferParseObjectInfo& bbox1, const NvDsInferParseObjectInfo& bbox2) {
        float area1 = bbox1.width * bbox1.height;
        float area2 = bbox2.width * bbox2.height;
        
        // Calculate area ratio and return similarity score
        float ratio = std::min(area1, area2) / std::max(area1, area2);
        return ratio; // Normalize to [0, 1]
    }
    
    // Assignment cost calculation using Hungarian algorithm
    // Enhanced version with velocity similarity and size difference
    void assignDetectionsToTracks(
        std::vector<TrackedObject>& tracks,
        std::vector<NvDsInferParseObjectInfo>& detections,
        std::vector<std::pair<int, int>>& assignments,
        std::vector<int>& unassignedTracks,
        std::vector<int>& unassignedDetections) {
        
        int numTracks = tracks.size();
        int numDetections = detections.size();
        
        if (numTracks == 0 || numDetections == 0) {
            // No tracks or detections to assign
            for (int i = 0; i < numDetections; i++) {
                unassignedDetections.push_back(i);
            }
            return;
        }
        
        // Create cost matrix
        cv::Mat costMatrix(numTracks, numDetections, CV_32F);
        
        for (int i = 0; i < numTracks; i++) {
            for (int j = 0; j < numDetections; j++) {
                // Calculate various similarity metrics
                float iou = computeIoU(tracks[i].bbox, detections[j]);
                float distance = calculateCentroidDistance(tracks[i].centroid, 
                                                         cv::Point2f(detections[j].left + detections[j].width/2, 
                                                                   detections[j].top + detections[j].height/2));
                float velocitySimilarity = calculateVelocitySimilarity(tracks[i], detections[j]);
                float sizeSimilarity = calculateSizeDifference(tracks[i].bbox, detections[j]);
                
                // Check class match if class-agnostic tracking is disabled
                float classMatch = 1.0f;
                if (!ENABLE_CLASS_AGNOSTIC_TRACKING && tracks[i].bbox.classId != detections[j].classId) {
                    classMatch = 0.0f;
                }
                
                // Calculate final similarity score with weighted factors
                float similarity = (iou * 0.5f) + (velocitySimilarity * 0.2f) + 
                                 (sizeSimilarity * 0.2f) + (classMatch * 0.1f);
                
                // Normalize distance and convert to similarity
                float distanceSimilarity = distance < TRACKING_DISTANCE_THRESHOLD ? 
                                         1.0f - (distance / TRACKING_DISTANCE_THRESHOLD) : 0.0f;
                
                // Combine similarity scores with distance factor
                similarity = (similarity * 0.7f) + (distanceSimilarity * 0.3f);
                
                // Cost is inverse of similarity (lower cost = better match)
                costMatrix.at<float>(i, j) = 1.0f - similarity;
            }
        }
        
        // Find assignments using Hungarian algorithm
        cv::Mat assignmentMatrix;
        cv::extractChannel(cv::optimalAssignment(costMatrix), assignmentMatrix, 0);
        
        // Process assignments
        std::vector<bool> assignedTracks(numTracks, false);
        std::vector<bool> assignedDetectionsLocal(numDetections, false);
        
        for (int i = 0; i < numTracks; i++) {
            int detIdx = assignmentMatrix.at<int>(i);
            if (detIdx < numDetections) {
                float cost = costMatrix.at<float>(i, detIdx);
                // Only accept assignments with low cost
                if (cost < 1.5f) { // Threshold based on our cost calculation
                    assignments.emplace_back(i, detIdx);
                    assignedTracks[i] = true;
                    assignedDetectionsLocal[detIdx] = true;
                }
            }
        }
        
        // Find unassigned tracks and detections
        for (int i = 0; i < numTracks; i++) {
            if (!assignedTracks[i]) {
                unassignedTracks.push_back(i);
            }
        }
        
        for (int i = 0; i < numDetections; i++) {
            if (!assignedDetectionsLocal[i]) {
                unassignedDetections.push_back(i);
            }
        }
    }
    
public:
    ObjectTracker() : nextId(0) {}
    
    void trackObjects(std::vector<NvDsInferParseObjectInfo>& detections) {
        std::lock_guard<std::mutex> lock(trackerMutex);
        
        // Start performance measurement
        g_profiler.startMeasurement("ObjectTracker::trackObjects");
        
        // Mark all objects as not updated and predict their next positions
        for (auto& obj : trackedObjects) {
            obj.incrementAge();
        }
        
        // Filter out inactive tracks for assignment
        std::vector<TrackedObject> activeTracks;
        std::vector<int> activeTrackIndices;
        
        for (size_t i = 0; i < trackedObjects.size(); i++) {
            if (trackedObjects[i].active) {
                activeTracks.push_back(trackedObjects[i]);
                activeTrackIndices.push_back(i);
            }
        }
        
        // Assign detections to existing tracks using Hungarian algorithm
        std::vector<std::pair<int, int>> assignments;
        std::vector<int> unassignedTracks;
        std::vector<int> unassignedDetections;
        cv::Mat costMatrix;
        
        // Modified to handle costMatrix locally
        int numTracks = activeTracks.size();
        int numDetectionsLocal = detections.size();
        
        if (numTracks > 0 && numDetectionsLocal > 0) {
            // Create cost matrix
            costMatrix.create(numTracks, numDetectionsLocal, CV_32F);
            
            for (int i = 0; i < numTracks; i++) {
                for (int j = 0; j < numDetectionsLocal; j++) {
                    // Calculate various similarity metrics
                    float iou = computeIoU(activeTracks[i].bbox, detections[j]);
                    float distance = calculateCentroidDistance(activeTracks[i].centroid, 
                                                             cv::Point2f(detections[j].left + detections[j].width/2, 
                                                                       detections[j].top + detections[j].height/2));
                    float velocitySimilarity = calculateVelocitySimilarity(activeTracks[i], detections[j]);
                    float sizeSimilarity = calculateSizeDifference(activeTracks[i].bbox, detections[j]);
                    
                    // Check class match if class-agnostic tracking is disabled
                    float classMatch = 1.0f;
                    if (!ENABLE_CLASS_AGNOSTIC_TRACKING && activeTracks[i].bbox.classId != detections[j].classId) {
                        classMatch = 0.0f;
                    }
                    
                    // Calculate final similarity score with weighted factors
                    float similarity = (iou * 0.5f) + (velocitySimilarity * 0.2f) + 
                                     (sizeSimilarity * 0.2f) + (classMatch * 0.1f);
                    
                    // Normalize distance and convert to similarity
                    float distanceSimilarity = distance < TRACKING_DISTANCE_THRESHOLD ? 
                                             1.0f - (distance / TRACKING_DISTANCE_THRESHOLD) : 0.0f;
                    
                    // Combine similarity scores with distance factor
                    similarity = (similarity * 0.7f) + (distanceSimilarity * 0.3f);
                    
                    // Cost is inverse of similarity (lower cost = better match)
                    costMatrix.at<float>(i, j) = 1.0f - similarity;
                }
            }
            
            // Find assignments using Hungarian algorithm
            cv::Mat assignmentMatrix;
            cv::extractChannel(cv::optimalAssignment(costMatrix), assignmentMatrix, 0);
            
            // Process assignments
            std::vector<bool> assignedTracks(numTracks, false);
            std::vector<bool> assignedDetectionsLocal(numDetectionsLocal, false);
            
            for (int i = 0; i < numTracks; i++) {
                int detIdx = assignmentMatrix.at<int>(i);
                if (detIdx < numDetectionsLocal) {
                    float cost = costMatrix.at<float>(i, detIdx);
                    // Only accept assignments with low cost
                    if (cost < 1.5f) { // Threshold based on our cost calculation
                        assignments.emplace_back(i, detIdx);
                        assignedTracks[i] = true;
                        assignedDetectionsLocal[detIdx] = true;
                    }
                }
            }
            
            // Find unassigned tracks and detections
            for (int i = 0; i < numTracks; i++) {
                if (!assignedTracks[i]) {
                    unassignedTracks.push_back(i);
                }
            }
            
            for (int i = 0; i < numDetectionsLocal; i++) {
                if (!assignedDetectionsLocal[i]) {
                    unassignedDetections.push_back(i);
                }
            }
        } else {
            // No tracks or detections to assign
            for (int i = 0; i < numDetectionsLocal; i++) {
                unassignedDetections.push_back(i);
            }
        }
        
        // Update assigned tracks with new detections
        for (const auto& assignment : assignments) {
            int trackIdx = activeTrackIndices[assignment.first];
            int detIdx = assignment.second;
            
            trackedObjects[trackIdx].update(detections[detIdx]);
            
            // Update tracking statistics
            trackingStats.successfulAssignments++;
            
            // Record assignment cost if costMatrix exists
            if (!costMatrix.empty()) {
                float cost = costMatrix.at<float>(assignment.first, assignment.second);
                trackingStats.assignmentCostHistory.push_back(cost);
                
                // Update average assignment cost
                trackingStats.averageAssignmentCost = 
                    (trackingStats.averageAssignmentCost * (trackingStats.assignmentCostHistory.size() - 1) + cost) / 
                    trackingStats.assignmentCostHistory.size();
            }
        }
        
        // Handle unassigned tracks
        for (int trackIdx : unassignedTracks) {
            trackingStats.failedAssignments++;
        }
        
        // Create new tracks for unassigned detections
        for (int detIdx : unassignedDetections) {
            trackedObjects.emplace_back(nextId++, detections[detIdx]);
            
            // Update tracking statistics
            trackingStats.objectsCreated++;
            trackingStats.totalObjectsTracked++;
            
            // Update class distribution
            int classId = detections[detIdx].classId;
            trackingStats.classDistribution[classId]++;
        }
        
        // Remove inactive tracks
        trackedObjects.erase(
            std::remove_if(trackedObjects.begin(), trackedObjects.end(),
                [](const TrackedObject& obj) {
                    return !obj.active;
                }),
            trackedObjects.end());
        
        // Limit the number of tracked objects
        if (trackedObjects.size() > MAX_OBJECTS_TRACKED) {
            std::sort(trackedObjects.begin(), trackedObjects.end(),
                [](const TrackedObject& a, const TrackedObject& b) {
                    // Prioritize tracks with recent updates and longer lifetime
                    if (a.timeSinceUpdate == b.timeSinceUpdate) {
                        return a.lifetime > b.lifetime;
                    }
                    return a.timeSinceUpdate < b.timeSinceUpdate;
                });
            trackedObjects.resize(MAX_OBJECTS_TRACKED);
        }
        
        LOG_DEBUG("Object tracking completed. Active tracks: " << trackedObjects.size());
        
        // End performance measurement
        g_profiler.endMeasurement("ObjectTracker::trackObjects");
    }
    
    std::vector<TrackedObject> getTrackedObjects() const {
        std::lock_guard<std::mutex> lock(trackerMutex);
        return trackedObjects;
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(trackerMutex);
        trackedObjects.clear();
        nextId = 0;
    }
    
    // Get tracking statistics
    std::unordered_map<std::string, int> getTrackingStatistics() const {
        std::lock_guard<std::mutex> lock(trackerMutex);
        
        std::unordered_map<std::string, int> stats;
        stats["totalTrackedObjects"] = trackedObjects.size();
        
        int activeObjects = 0;
        int inactiveObjects = 0;
        
        for (const auto& obj : trackedObjects) {
            if (obj.active) {
                activeObjects++;
            } else {
                inactiveObjects++;
            }
        }
        
        stats["activeObjects"] = activeObjects;
        stats["inactiveObjects"] = inactiveObjects;
        
        return stats;
    }
    
    // Get detailed tracking statistics including trackingStats data
    std::unordered_map<std::string, double> getDetailedTrackingStatistics() const {
        std::lock_guard<std::mutex> lock(trackerMutex);
        
        std::unordered_map<std::string, double> stats;
        
        // Basic tracking statistics
        stats["totalObjectsTracked"] = trackingStats.totalObjectsTracked;
        stats["objectsCreated"] = trackingStats.objectsCreated;
        stats["objectsLost"] = trackingStats.objectsLost;
        stats["successfulAssignments"] = trackingStats.successfulAssignments;
        stats["failedAssignments"] = trackingStats.failedAssignments;
        stats["averageTrackLifetime"] = trackingStats.averageTrackLifetime;
        stats["averageAssignmentCost"] = trackingStats.averageAssignmentCost;
        
        // Calculate current tracking time
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - trackingStats.startTime).count();
        stats["trackingTimeSeconds"] = duration;
        
        // Calculate assignment success rate
        double totalAssignments = trackingStats.successfulAssignments + trackingStats.failedAssignments;
        stats["assignmentSuccessRate"] = totalAssignments > 0 ? 
            (trackingStats.successfulAssignments / totalAssignments) * 100.0 : 0.0;
        
        return stats;
    }
    
    // Get class distribution statistics
    std::unordered_map<int, int> getClassDistribution() const {
        std::lock_guard<std::mutex> lock(trackerMutex);
        return trackingStats.classDistribution;
    }
    
    // Get assignment cost history
    std::vector<double> getAssignmentCostHistory() const {
        std::lock_guard<std::mutex> lock(trackerMutex);
        return trackingStats.assignmentCostHistory;
    }
};

// Advanced bounding box post-processor
class BBoxPostProcessor {
public:
    // Apply soft NMS to refine bounding boxes
    static std::vector<NvDsInferParseObjectInfo> applySoftNMS(
        const std::vector<NvDsInferParseObjectInfo>& bboxes, float sigma = 0.5f, float scoreThreshold = 0.001f) {
        std::vector<NvDsInferParseObjectInfo> result;
        std::vector<NvDsInferParseObjectInfo> tempBoxes = bboxes;
        
        while (!tempBoxes.empty()) {
            // Find the box with highest confidence
            auto maxIt = std::max_element(tempBoxes.begin(), tempBoxes.end(),
                [](const NvDsInferParseObjectInfo& a, const NvDsInferParseObjectInfo& b) {
                    return a.detectionConfidence < b.detectionConfidence;
                });
            
            NvDsInferParseObjectInfo maxBox = *maxIt;
            result.push_back(maxBox);
            tempBoxes.erase(maxIt);
            
            // Apply soft suppression to remaining boxes
            for (auto& box : tempBoxes) {
                float iou = computeIoU(maxBox, box);
                float weight = std::exp(-(iou * iou) / sigma);
                box.detectionConfidence *= weight;
            }
            
            // Remove boxes with confidence below threshold
            tempBoxes.erase(
                std::remove_if(tempBoxes.begin(), tempBoxes.end(),
                    [scoreThreshold](const NvDsInferParseObjectInfo& box) {
                        return box.detectionConfidence < scoreThreshold;
                    }),
                tempBoxes.end());
        }
        
        return result;
    }
    
    // Apply bounding box clamping to network boundaries
    static void clampBBoxes(std::vector<NvDsInferParseObjectInfo>& bboxes, uint netW, uint netH) {
        for (auto& bbox : bboxes) {
            bbox.left = clamp(bbox.left, 0.0f, static_cast<float>(netW));
            bbox.top = clamp(bbox.top, 0.0f, static_cast<float>(netH));
            bbox.width = clamp(bbox.width, 1.0f, static_cast<float>(netW) - bbox.left);
            bbox.height = clamp(bbox.height, 1.0f, static_cast<float>(netH) - bbox.top);
        }
    }
    
    // Apply bounding box normalization for consistent tracking
    static void normalizeBBoxes(std::vector<NvDsInferParseObjectInfo>& bboxes, uint netW, uint netH) {
        for (auto& bbox : bboxes) {
            bbox.left /= static_cast<float>(netW);
            bbox.top /= static_cast<float>(netH);
            bbox.width /= static_cast<float>(netW);
            bbox.height /= static_cast<float>(netH);
        }
    }
    
    // Denormalize bounding boxes back to original dimensions
    static void denormalizeBBoxes(std::vector<NvDsInferParseObjectInfo>& bboxes, uint netW, uint netH) {
        for (auto& bbox : bboxes) {
            bbox.left *= static_cast<float>(netW);
            bbox.top *= static_cast<float>(netH);
            bbox.width *= static_cast<float>(netW);
            bbox.height *= static_cast<float>(netH);
        }
    }
};

// Kalman filter implementation for object tracking with acceleration and adaptive noise
class KalmanFilter {
private:
    cv::KalmanFilter kf;
    cv::Mat state;  // [x, y, vx, vy, ax, ay] - position, velocity, acceleration
    cv::Mat measurement;  // [x, y] - measurement
    float q;  // Base process noise covariance
    float r;  // Base measurement noise covariance
    float adaptiveQ;  // Adaptive process noise covariance
    float adaptiveR;  // Adaptive measurement noise covariance
    bool initialized;  // Whether the filter has been initialized
    bool useAcceleration;  // Whether to use acceleration state variables
    
public:
    KalmanFilter(float processNoise = KALMAN_Q, float measurementNoise = KALMAN_R, bool enableAcceleration = true)
        : q(processNoise), r(measurementNoise), initialized(false), useAcceleration(enableAcceleration) {
        // Initialize Kalman filter with 6 or 4 state variables (x, y, vx, vy, ax, ay) or (x, y, vx, vy)
        // and 2 measurement variables (x, y)
        int stateSize = useAcceleration ? 6 : 4;
        kf.init(stateSize, 2, 0);
        
        // State transition matrix: x(k+1) = F * x(k)
        if (useAcceleration) {
            // With acceleration - 6 state variables
            kf.transitionMatrix = (cv::Mat_<float>(6, 6) <<
                1, 0, 1, 0, 0.5, 0,    // x + vx + 0.5*ax
                0, 1, 0, 1, 0, 0.5,    // y + vy + 0.5*ay
                0, 0, 1, 0, 1, 0,      // vx + ax
                0, 0, 0, 1, 0, 1,      // vy + ay
                0, 0, 0, 0, 1, 0,      // ax remains (constant acceleration model)
                0, 0, 0, 0, 0, 1       // ay remains (constant acceleration model)
            );
        } else {
            // Without acceleration - 4 state variables
            kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                1, 0, 1, 0,  // x + vx
                0, 1, 0, 1,  // y + vy
                0, 0, 1, 0,  // vx remains
                0, 0, 0, 1   // vy remains
            );
        }
        
        // Measurement matrix: z(k) = H * x(k)
        if (useAcceleration) {
            // 6 state variables -> 2 measurements
            kf.measurementMatrix = (cv::Mat_<float>(2, 6) <<
                1, 0, 0, 0, 0, 0,  // measure x
                0, 1, 0, 0, 0, 0   // measure y
            );
        } else {
            // 4 state variables -> 2 measurements
            kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
                1, 0, 0, 0,  // measure x
                0, 1, 0, 0   // measure y
            );
        }
        
        // Initialize with base noise values
        adaptiveQ = q;
        adaptiveR = r;
        
        // Process noise covariance matrix
        kf.processNoiseCov = cv::Mat::eye(stateSize, stateSize, CV_32F) * adaptiveQ;
        
        // Measurement noise covariance matrix
        kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * adaptiveR;
        
        // Error covariance matrix
        kf.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_32F);
        
        // State matrix
        state = cv::Mat::zeros(stateSize, 1, CV_32F);
        
        // Measurement matrix
        measurement = cv::Mat::zeros(2, 1, CV_32F);
    }
    
    // Initialize the filter with the first measurement
    void initialize(const cv::Point2f& measurement) {
        state.at<float>(0) = measurement.x;
        state.at<float>(1) = measurement.y;
        state.at<float>(2) = 0;  // Initial velocity x
        state.at<float>(3) = 0;  // Initial velocity y
        
        if (useAcceleration) {
            state.at<float>(4) = 0;  // Initial acceleration x
            state.at<float>(5) = 0;  // Initial acceleration y
        }
        
        kf.statePost = state;
        initialized = true;
    }
    
    // Predict the next state
    cv::Point2f predict() {
        if (!initialized) {
            return cv::Point2f(0, 0);
        }
        
        cv::Mat prediction = kf.predict();
        return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }
    
    // Update the filter with a new measurement
    cv::Point2f update(const cv::Point2f& measurement) {
        if (!initialized) {
            initialize(measurement);
            return measurement;
        }
        
        // Set measurement
        this->measurement.at<float>(0) = measurement.x;
        this->measurement.at<float>(1) = measurement.y;
        
        // Calculate measurement residual before update
        cv::Mat prediction = kf.predict();
        cv::Mat residual = this->measurement - kf.measurementMatrix * prediction;
        float residualNorm = cv::norm(residual);
        
        // Adaptive noise adjustment based on residual
        // Larger residual means higher uncertainty in measurement or prediction
        float adaptiveFactor = std::min(std::max(residualNorm / 100.0f, 0.5f), 2.0f);
        adaptiveQ = q * adaptiveFactor;
        adaptiveR = r / adaptiveFactor;
        
        // Update noise covariance matrices
        int stateSize = useAcceleration ? 6 : 4;
        kf.processNoiseCov = cv::Mat::eye(stateSize, stateSize, CV_32F) * adaptiveQ;
        kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * adaptiveR;
        
        // Update the filter
        cv::Mat estimated = kf.correct(this->measurement);
        
        return cv::Point2f(estimated.at<float>(0), estimated.at<float>(1));
    }
    
    // Get the current estimated state
    cv::Point2f getCurrentState() const {
        if (!initialized) {
            return cv::Point2f(0, 0);
        }
        return cv::Point2f(kf.statePost.at<float>(0), kf.statePost.at<float>(1));
    }
    
    // Get the current estimated velocity
    cv::Point2f getCurrentVelocity() const {
        if (!initialized) {
            return cv::Point2f(0, 0);
        }
        return cv::Point2f(kf.statePost.at<float>(2), kf.statePost.at<float>(3));
    }
    
    // Get the current estimated acceleration
    cv::Point2f getCurrentAcceleration() const {
        if (!initialized || !useAcceleration) {
            return cv::Point2f(0, 0);
        }
        return cv::Point2f(kf.statePost.at<float>(4), kf.statePost.at<float>(5));
    }
    
    // Get the current adaptive process noise covariance
    float getCurrentAdaptiveQ() const {
        return adaptiveQ;
    }
    
    // Get the current adaptive measurement noise covariance
    float getCurrentAdaptiveR() const {
        return adaptiveR;
    }
    
    // Check if acceleration is enabled
    bool isAccelerationEnabled() const {
        return useAcceleration;
    }
    
    // Reset the filter
    void reset() {
        initialized = false;
    }
    
    // Check if the filter is initialized
    bool isInitialized() const {
        return initialized;
    }
};

// Global object tracker instance
static ObjectTracker g_objectTracker;

static std::vector<NvDsInferParseObjectInfo>
parseYoloV5BBox(const NvDsInferLayerInfo& feat, const uint numOutputClasses, const uint& netW,
    const uint& netH)
{
    // Start performance measurement
    g_profiler.startMeasurement("parseYoloV5BBox");
    
    // Load config if not already loaded
    static std::once_flag configLoadedFlag;
    std::call_once(configLoadedFlag, []() {
        g_configManager.loadConfig();
    });
    
    // Get parameters from config or use defaults
    int numClasses = g_configManager.getValue("NUM_CLASSES_YOLO", NUM_CLASSES_YOLO);
    float nmsThreshold = g_configManager.getValue("NMS_THRESHOLD", NMS_THRESHOLD);
    float clsThreshold = g_configManager.getValue("CLS_THRESHOLD", CLS_THRESHOLD);
    int maxWorkerThreads = g_configManager.getValue("MAX_WORKER_THREADS", MAX_WORKER_THREADS);
    bool useSoftNMS = g_configManager.getValue("USE_SOFT_NMS", false);
    
    std::vector<std::vector<NvDsInferParseObjectInfo>> binfo;
    binfo.resize(numClasses);

    const float* detections = (const float*)feat.buffer;
    auto numBBoxes = feat.inferDims.d[0];
    const int numBBoxCells = feat.inferDims.d[1];

    LOG_DEBUG("Processing " << numBBoxes << " bounding boxes from feature map");
    
    // Process all bounding boxes
    for (uint b = 0; b < numBBoxes; ++b)
    {
        const float bx = detections[b * numBBoxCells + 0];
        const float by = detections[b * numBBoxCells + 1];
        const float bw = detections[b * numBBoxCells + 2];
        const float bh = detections[b * numBBoxCells + 3];
        const float objectness = detections[b * numBBoxCells + 4];

        float maxProb = 0.0f;
        int maxIndex = -1;

        // Find the class with highest probability
        for (uint i = 0; i < numOutputClasses; ++i)
        {
            float prob = detections[b * numBBoxCells + (5 + i)];
            if (prob > maxProb) {
                maxProb = prob;
                maxIndex = i;
            }
        }
        
        // Apply objectness score
        maxProb = objectness * maxProb;
        
        // Add to bounding box list if confidence is above threshold
        if(maxProb > clsThreshold) {
            addBBoxProposal(bx, by, bw, bh, netW, netH, maxIndex, maxProb, binfo[maxIndex]);
        }
    }
    
    LOG_DEBUG("Bounding box parsing completed");
    
    // Create parallel NMS processor
    ParallelNMSProcessor nmsProcessor(maxWorkerThreads);
    std::vector<NvDsInferParseObjectInfo> objects;
    std::mutex resultMutex;
    
    // Submit NMS tasks for each class
    for(int cls_id = 0; cls_id < numClasses; ++cls_id) {
        if (!binfo[cls_id].empty()) {
            nmsProcessor.processClass(cls_id, binfo[cls_id], nmsThreshold, objects, resultMutex);
        }
    }
    
    // Wait for all NMS tasks to complete
    nmsProcessor.waitForCompletion();
    
    // Apply additional post-processing if enabled
    if (useSoftNMS) {
        LOG_DEBUG("Applying Soft NMS");
        objects = BBoxPostProcessor::applySoftNMS(objects);
    }
    
    // Ensure bounding boxes are within network boundaries
    BBoxPostProcessor::clampBBoxes(objects, netW, netH);
    
    LOG_DEBUG("Final object count after post-processing: " << objects.size());
    
    // End performance measurement
    g_profiler.endMeasurement("parseYoloV5BBox");
    
    return objects;
}

// Class for managing object detection statistics
class DetectionStatistics {
private:
    std::unordered_map<int, int> classCountMap;
    std::mutex statsMutex;
    int totalDetections;
    std::chrono::time_point<std::chrono::system_clock> startTime;
    
public:
    DetectionStatistics() : totalDetections(0) {
        startTime = std::chrono::system_clock::now();
    }
    
    void addDetection(int classId) {
        std::lock_guard<std::mutex> lock(statsMutex);
        classCountMap[classId]++;
        totalDetections++;
    }
    
    void updateStatistics(const std::vector<NvDsInferParseObjectInfo>& objects) {
        for (const auto& obj : objects) {
            addDetection(obj.classId);
        }
    }
    
    void printStatistics() {
        std::lock_guard<std::mutex> lock(statsMutex);
        
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        
        LOG_INFO("=== Detection Statistics ===");
        LOG_INFO("Total detections: " << totalDetections);
        LOG_INFO("Running time: " << duration << " seconds");
        
        if (duration > 0) {
            LOG_INFO("Detection rate: " << std::fixed << std::setprecision(2) 
                     << static_cast<double>(totalDetections) / duration << " objects/second");
        }
        
        LOG_INFO("Class distribution:");
        for (const auto& [classId, count] : classCountMap) {
            LOG_INFO("  Class " << classId << ": " << count << " objects");
        }
        
        LOG_INFO("========================");
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(statsMutex);
        classCountMap.clear();
        totalDetections = 0;
        startTime = std::chrono::system_clock::now();
    }
};

// Global detection statistics instance
static DetectionStatistics g_detectionStats;

static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Start performance measurement
    g_profiler.startMeasurement("NvDsInferParseYoloV5");
    
    LOG_DEBUG("Starting YOLOv5 inference parsing");
    LOG_DEBUG("Number of output layers: " << outputLayersInfo.size());
    LOG_DEBUG("Network input dimensions: " << networkInfo.width << "x" << networkInfo.height);
    
    // Validate input parameters
    try {
        if (outputLayersInfo.empty()) {
            LOG_ERROR("No output layers provided for parsing");
            g_profiler.endMeasurement("NvDsInferParseYoloV5");
            return false;
        }
        
        if (networkInfo.width <= 0 || networkInfo.height <= 0) {
            LOG_ERROR("Invalid network dimensions: " << networkInfo.width << "x" << networkInfo.height);
            g_profiler.endMeasurement("NvDsInferParseYoloV5");
            return false;
        }
        
        // Check for class count mismatch
        int configuredClasses = detectionParams.numClassesConfigured;
        int networkClasses = NUM_CLASSES_YOLO;
        
        if (networkClasses != configuredClasses) {
            LOG_WARNING("Class count mismatch: Configured=" << configuredClasses 
                       << ", Network=" << networkClasses);
        }
        
        // Get output layer (YOLOv5 typically has a single output layer)
        int outputLayerIndex = g_configManager.getValue("OUTPUT_LAYER_INDEX", 3);
        
        if (outputLayerIndex >= outputLayersInfo.size()) {
            LOG_ERROR("Output layer index " << outputLayerIndex << " out of range (0-" 
                     << outputLayersInfo.size()-1 << ")");
            g_profiler.endMeasurement("NvDsInferParseYoloV5");
            return false;
        }
        
        const NvDsInferLayerInfo &layer = outputLayersInfo[outputLayerIndex];
        
        // Validate layer dimensions
        if (layer.inferDims.numDims != 2) {
            LOG_ERROR("Expected output layer with 2 dimensions, got " << layer.inferDims.numDims);
            g_profiler.endMeasurement("NvDsInferParseYoloV5");
            return false;
        }
        
        LOG_DEBUG("Output layer dimensions: [" << layer.inferDims.d[0] << ", " 
                 << layer.inferDims.d[1] << "]");
        
        // Parse bounding boxes from the output feature map
        std::vector<NvDsInferParseObjectInfo> objects = parseYoloV5BBox(
            layer, NUM_CLASSES_YOLO, networkInfo.width, networkInfo.height);
        
        // Apply object tracking
        g_objectTracker.trackObjects(objects);
        
        // Update detection statistics
        g_detectionStats.updateStatistics(objects);
        
        // Log tracking information
        auto trackedObjects = g_objectTracker.getTrackedObjects();
        LOG_DEBUG("Currently tracking " << trackedObjects.size() << " objects");
        
        // Assign results to output
        objectList.swap(objects);  // More efficient than assignment
        
        LOG_INFO("Inference parsing completed successfully. Detected " << objectList.size() << " objects");
        
        // Print performance statistics periodically
        static int parseCount = 0;
        if (++parseCount % 100 == 0) {
            g_profiler.printStatistics();
            g_detectionStats.printStatistics();
        }
        
        // End performance measurement
        g_profiler.endMeasurement("NvDsInferParseYoloV5");
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in NvDsInferParseYoloV5: " << e.what());
        g_profiler.endMeasurement("NvDsInferParseYoloV5");
        return false;
    } catch (...) {
        LOG_CRITICAL("Unknown exception in NvDsInferParseYoloV5");
        g_profiler.endMeasurement("NvDsInferParseYoloV5");
        return false;
    }
}

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV5 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
