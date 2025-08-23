/*!
\file EvaluateSceneFlow.h
\brief Scene Flow evaluation module - Fixed version
\author Gang Wang, Durham University
*/

#pragma once

#include "SceneFlowReconstructor.h"
#include "FeatureMatcher.h"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <map>
using namespace cv;

// 🆕 标准化场景流评估指标结构体
struct SceneFlowMetrics
{
    double EPE3d;    // 平均3D端点误差 (米)
    double AccS;     // 严格准确率 (EPE<0.05m 或 相对误差<5%)
    double AccR;     // 宽松准确率 (EPE<0.1m 或 相对误差<10%)
    double Outlier;  // 离群值比例 (EPE>0.3m 或 相对误差>10%)
    int valid_count; // 有效像素数
    double time_ms;  // 计算时间(毫秒)

    // 构造函数
    SceneFlowMetrics() : EPE3d(0.0), AccS(0.0), AccR(0.0), Outlier(0.0), valid_count(0), time_ms(0.0) {}
};

class EvaluateSceneFlow
{
public:
    EvaluateSceneFlow();

    /**
     * @brief 统一的场景流评估接口 - 支持单帧和批量处理
     * @param method 光流算法名称 (e.g., "degraf_flow_rlof", "farneback", "tvl1")
     * @param display_images 是否显示可视化窗口和保存图像
     * @param image_indices 图像索引数组（单帧传{i}，批量传{i1,i2,...}）
     * @return 评估结果数组
     */
    std::vector<SceneFlowMetrics> runEvaluation(
        const std::string &method,
        bool display_images,
        const std::vector<int> &image_indices);

    /**
     * @brief 便利重载 - 保持向后兼容的单帧接口
     * @param method 光流算法名称
     * @param display_images 是否显示可视化
     * @param image_no KITTI图像序列号
     * @return 单帧评估结果
     */
    SceneFlowMetrics runEvaluation(
        const std::string &method,
        bool display_images,
        int image_no);

    // 获取所有结果（用于统计）
    const std::vector<SceneFlowMetrics> &getAllResults() const { return all_results_; }

    // 清空结果
    void clearResults() { all_results_.clear(); }

    // 新增：生成场景流对比表格
    void exportSceneFlowComparisonCSV(
        const std::string& csv_path,
        const std::map<std::string, std::vector<SceneFlowMetrics>>& method_results);
    
    // 新增：生成4宫格可视化图
    void generateSceneFlow4PanelVisualization(
        const std::string& method_name,
        int image_no,
        const cv::Mat& original_image,
        const cv::Mat& pred_scene_flow,
        const cv::Mat& gt_scene_flow,
        const std::string& output_path
    );


private:
    // 🔧 修复：将成员变量移到private区域
    std::vector<SceneFlowMetrics> all_results_; // 存储所有评估结果

    // 🔒 核心计算函数 - 私有化
    SceneFlowMetrics calculateStandardMetrics(const cv::Mat &pred_scene_flow,
                                              const cv::Mat &gt_scene_flow);
    SceneFlowMetrics evaluateSingleFrame(const cv::Mat &pred_scene_flow,
                                         const cv::Mat &gt_scene_flow,
                                         bool verbose = true);
    void writeMetricsToCSV(const SceneFlowMetrics &metrics,
                           const std::string &method,
                           int image_no,
                           const std::string &csv_path);
    // 新增：场景流转换为彩色可视化图
    cv::Mat sceneFlowToColorMap(const cv::Mat& scene_flow_3d);
    
    // 新增：计算场景流误差热图
    cv::Mat computeSceneFlowErrorMap(const cv::Mat& pred_sf, const cv::Mat& gt_sf);

    // 新增：视差稠密化
    cv::Mat densifyDisparity(const cv::Mat& sparse_disp);
    
    // 新增：场景流稠密化  
    cv::Mat densifySceneFlow(const cv::Mat& sparse_sf);
    
    // 新增：HSV转RGB（KITTI标准）
    void hsvToRgb(float h, float s, float v, float &r, float &g, float &b);

};