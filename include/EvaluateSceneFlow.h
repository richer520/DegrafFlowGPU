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
     * @brief Main KITTI Scene Flow evaluation process (comprehensive evaluation)
     * @param method Optical flow algorithm name (e.g., "degraf_flow_rlof", "farneback", "tvl1")
     * @param display_images Whether to display visualization windows and save images
     * @param image_no KITTI image sequence number (e.g., 0 for 000000_10.png)
     * @return SceneFlowMetrics evaluation results
     */
    SceneFlowMetrics runEvaluation(const std::string &method, bool display_images, int image_no);
    // 获取所有结果（用于统计）
    const std::vector<SceneFlowMetrics> &getAllResults() const { return all_results_; }

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
};