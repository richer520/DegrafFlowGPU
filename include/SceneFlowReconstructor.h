/*!
\file SceneFlowReconstructor.h
\brief Scene Flow reconstruction from optical flow and disparity maps - Core computation only
\author Gang Wang, Durham University
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class SceneFlowReconstructor
{
public:
    // Constructor with custom camera parameters
    SceneFlowReconstructor(float fx, float fy, float cx, float cy, float baseline);

    // Default constructor - requires setCameraParameters() to be called before use
    SceneFlowReconstructor();

    // ✅ 新增：设置相机内参函数
    void setCameraParameters(float fx, float fy, float cx, float cy, float baseline);

    // 🎯 Core function: compute 3D scene flow from 2D optical flow and disparity maps
    cv::Mat computeSceneFlow(const cv::Mat &flow,
                             const cv::Mat &disp0,
                             const cv::Mat &disp1 = cv::Mat());

    // Getter functions for camera parameters (for visualization modules to use)
    float getFx() const { return fx_; }
    float getFy() const { return fy_; }
    float getCx() const { return cx_; }
    float getCy() const { return cy_; }
    float getBaseline() const { return baseline_; }

    // ✅ 新增：检查相机参数是否已设置
    bool isCameraParametersSet() const { return camera_params_set_; }

private:
    // Camera intrinsic parameters
    float fx_, fy_, cx_, cy_, baseline_;

    // ✅ 新增：标记相机参数是否已设置
    bool camera_params_set_;

    // ✅ Core computation helper functions - kept here
    cv::Point3f reprojectTo3D(int u, int v, float disparity) const;
    bool isValidDisparity(float disparity) const;
    bool isValidPoint(const cv::Point3f &point) const;

    // Constants for validation
    static constexpr float MIN_DISPARITY = 0.1f;
    static constexpr float MAX_DISPARITY = 256.0f;
    static constexpr float MAX_DEPTH = 100.0f;
};