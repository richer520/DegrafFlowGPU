/*!
\file EvaluateSceneFlow.h
\brief Scene Flow evaluation module - 严格参考EvaluateOptFlow设计风格
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

using namespace cv;

class EvaluateSceneFlow
{
public:
    std::vector<double> stats_vector;
    std::vector<std::vector<double>> all_stats;

    EvaluateSceneFlow();

    /**
     * @brief 主评估流程（多区域mask/统计/可视化/写csv）
     * @param method 光流算法名
     * @param display_images 是否显示窗口
     * @param image_no 图像编号
     * @return int 0=成功，-1=失败
     */
    int runEvaluation(const std::string &method, bool display_images, int image_no);

    /**
     * @brief 计算一帧场景流误差统计（均值EPE、R0.1、R0.2、A90等）
     * @param pred_scene_flow 预测场景流
     * @param gt_scene_flow GT场景流
     * @param image 原始输入图像
     * @param region 区域名("all"/"discontinuities"/"untextured")
     * @param display_images 是否显示
     * @param stats_vector 输出统计结果
     */
    void calculateSceneFlowStats(const cv::Mat &pred_scene_flow,
                                 const cv::Mat &gt_scene_flow,
                                 const cv::Mat &image,
                                 const std::string &region,
                                 bool display_images,
                                 std::vector<double> &stats_vector);
};