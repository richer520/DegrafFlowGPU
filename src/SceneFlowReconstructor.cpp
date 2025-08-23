/*!
\file SceneFlowReconstructor.cpp
\brief Scene Flow reconstruction implementation - Core computation only
\author Gang Wang, Durham University
*/

#include "SceneFlowReconstructor.h"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// Constructor with custom camera parameters
/*
参数构造器，这里已经默认设置好相机内参。可以直接调用，具体传的默认参数在外部函数中设置。
fx, fy	相机的焦距（像素单位），X/Y 方向的尺度因子
cx, cy	相机主点（中心点）在图像中的位置
baseline	双目相机的基线（单位：米），左右相机之间的水平距离
*/
SceneFlowReconstructor::SceneFlowReconstructor(float fx, float fy, float cx, float cy, float baseline)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), camera_params_set_(true)
{
    // fx, fy, cx, cy, baseline are set directly
    cout << "SceneFlowReconstructor initialized with custom parameters:" << endl;
    cout << "fx=" << fx_ << ", fy=" << fy_ << ", cx=" << cx_ << ", cy=" << cy_
         << ", baseline=" << baseline_ << endl;
}

// Default constructor - requires setCameraParameters() to be called before use
// 默认构造函数，未设置相机内参，需要在使用前调用 setCameraParameters() 设置。
// 这里的默认参数是0，表示未设置。
// 需要在使用前调用 setCameraParameters() 设置相机内参。
SceneFlowReconstructor::SceneFlowReconstructor()
    : fx_(0), fy_(0), cx_(0), cy_(0), baseline_(0), camera_params_set_(false)
{
    cout << "SceneFlowReconstructor initialized. Please call setCameraParameters() before use." << endl;
}

// 设置相机内参函数
void SceneFlowReconstructor::setCameraParameters(float fx, float fy, float cx, float cy, float baseline)
{
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    baseline_ = baseline;
    camera_params_set_ = true;

    cout << "Camera parameters set:" << endl;
    cout << "fx=" << fx_ << ", fy=" << fy_ << ", cx=" << cx_ << ", cy=" << cy_
         << ", baseline=" << baseline_ << endl;
}

// Core function: compute 3D scene flow
/**
 * \brief 计算3D场景流
 * \param flow 2D光流图像，类型为CV_32FC2
 * \param disp0 第一个视差图，类型为CV_32F或CV_16UC1
 * \param disp1 第二个视差图（可选），类型为CV_32F或CV_16UC1
 * \return 返回计算得到的3D场景流，类型为CV_32FC3,每个像素是一个三维向量
 */
Mat SceneFlowReconstructor::computeSceneFlow(const Mat &flow,
                                             const Mat &disp0,
                                             const Mat &disp1)
{
    // Step 0 : 检查相机参数是否已设置
    // 如果调用者忘了设置 fx, fy, cx, cy, baseline，直接报错退出。因为接下来的 3D 投影必须依赖这些内参。
    if (!camera_params_set_)
    {
        cerr << "Error: Camera parameters not set! Please call setCameraParameters() first." << endl;
        return Mat();
    }

    cout << "Flow size: " << flow.cols << "x" << flow.rows << endl;
    cout << "Disp0 size: " << disp0.cols << "x" << disp0.rows << endl;
    cout << "Flow type: " << flow.type() << endl;
    cout << "Disp0 type: " << disp0.type() << endl;

    // Input validation 验证输入有效性。
    // 确保 flow 是二维光流图，且 disp0 与之大小一致，且不为空。否则视为无效输入。
    CV_Assert(!flow.empty() && flow.type() == CV_32FC2);
    CV_Assert(!disp0.empty());
    CV_Assert(flow.size() == disp0.size());

    cout << "Computing scene flow..." << endl;
    cout << "Flow dimensions: " << flow.cols << "x" << flow.rows << endl;
    cout << "Disparity dimensions: " << disp0.cols << "x" << disp0.rows << endl;

    // Step 1: Convert disparity maps to proper format
    Mat disp0_f32, disp1_f32;

    // 由于EvaluateSceneFlow中已经按KITTI标准转换过了，这里直接使用
    if (disp0.type() == CV_32F)
    {
        disp0_f32 = disp0.clone();
        cout << "Using pre-converted disparity0 as float32" << endl;
    }
    else if (disp0.type() == CV_16UC1)
    {
        // 这种情况不应该发生，因为输入已经预处理过
        cout << "Warning: Received raw KITTI format, converting..." << endl;
        disp0.convertTo(disp0_f32, CV_32F, 1.0 / 256.0);
    }
    else
    {
        cerr << "Error: Unsupported disparity format: " << disp0.type() << endl;
        return Mat();
    }

    bool use_temporal_disparity = !disp1.empty(); // 检查是否提供了第二个视差图
    if (use_temporal_disparity)
    {
        /**
         * \brief 如果提供了第二个视差图 disp1，检查其类型并转换为 CV_32F 格式。
         * \details 如果 disp1 是 CV_16UC1 格式，则转换为 CV_32F 格式并缩放。
         * 如果是 CV_32F 格式，则直接使用。
         * 如果类型不支持，则报错并返回空 Mat。
         */

        CV_Assert(disp1.size() == flow.size());
        if (disp1.type() == CV_32F)
        {
            disp1_f32 = disp1.clone();
        }
        else if (disp1.type() == CV_16UC1)
        {
            cout << "Warning: Received raw KITTI disp1 format, converting..." << endl;
            disp1.convertTo(disp1_f32, CV_32F, 1.0 / 256.0);
        }
        cout << "Using temporal disparity information." << endl;
    }
    else
    {
        cout << "Using single disparity map (constant depth assumption)." << endl;
    }

    // Step 2: Initialize scene flow output (3D motion vectors)
    // 构造一个与光流图同大小的图像，每个像素一个三维向量，初始为 0。
    Mat sceneFlow(flow.size(), CV_32FC3, Scalar::all(0));

    int valid_points = 0;                     // 统计有效点的数量
    int total_points = flow.rows * flow.cols; // 总点数

    // Step 3: Process each pixel
    // 遍历光流图的每个像素，计算对应的 3D 场景流。
    // 对每个像素 (u, v)，计算其在时间 t 和 t+1 的 3D 位置。
    // 使用光流向量和视差图来计算场景流。
    // flow.rows 和 flow.cols 分别是图像的高度和宽度。
    // (u,v) 是当前像素的坐标，flow.at<Point2f>(v, u) 获取该像素的光流向量。
    /**
     * 光流 flow + 视差 disp0 + disp1
     * ↓
     * P0 = reprojectTo3D(u, v, disp0)
     * P1 = reprojectTo3D(u+flow.x, v+flow.y, disp1)
     * ↓
     * scene_flow(u,v) = P1 - P0
     *
     */
    for (int v = 0; v < flow.rows; ++v)
    {
        for (int u = 0; u < flow.cols; ++u)
        {
            // Step 3.0: 获取光流向量
            // 获取当前像素的光流向量，类型为 Point2f。
            Point2f flow_vec = flow.at<Point2f>(v, u); // 获取当前像素的光流向量
            float disp_t0 = disp0_f32.at<float>(v, u); // 获取时间 t 的视差值 ，从 t 时刻的视差图中，取出位置 (u,v) 对应的 float 类型视差值，赋值给 disp_t0

            // Skip invalid disparity at time t
            if (!isValidDisparity(disp_t0)) // 检查视差值是否有效
            {
                continue; // 如果视差无效，跳过当前像素
            }

            // Step 3.1: Compute 3D position at time t
            Point3f P0 = reprojectTo3D(u, v, disp_t0); // 将像素坐标 (u, v) 和视差 disp_t0 重投影到 3D 空间
            // 检查重投影后的 3D 点是否有效
            if (!isValidPoint(P0)) // 如果重投影后的点无效，跳过当前像素
            {
                continue;
            }

            // Step 3.2: Calculate target pixel position using optical flow
            // cvRound 将光流向量 (u + flow_vec.x, v + flow_vec.y) 四舍五入到最近的整数。
            // flow_vec.x 和 flow_vec.y 分别是光流向量在 x 和 y 方向的分量。
            int u1 = cvRound(u + flow_vec.x); // 计算时间 t+1 的像素位置
            int v1 = cvRound(v + flow_vec.y); // 计算时间 t+1 的像素位置

            // Check bounds for target position
            if (u1 < 0 || u1 >= flow.cols || v1 < 0 || v1 >= flow.rows)
            {
                continue;
            }

            // Step 3.3: Get disparity at time t+1
            float disp_t1;
            if (use_temporal_disparity) // 如果提供了第二个视差图 disp1
            {
                disp_t1 = disp1_f32.at<float>(v1, u1); // 获取时间 t+1 的视差值
            }
            else
            {
                // Constant depth assumption: disp_t1 = disp_t0
                disp_t1 = disp_t0; // 如果没有提供第二个视差图，假设时间 t+1 的视差与时间 t 相同
            }

            if (!isValidDisparity(disp_t1)) // 检查时间 t+1 的视差值是否有效
            {
                continue; // 如果视差无效，跳过当前像素
            }

            // Step 3.4: Compute 3D position at time t+1
            Point3f P1 = reprojectTo3D(u1, v1, disp_t1); // 将时间 t+1 的像素坐标 (u1, v1) 和视差 disp_t1 重投影到 3D 空间
            if (!isValidPoint(P1))
            {
                continue;
            }

            // Step 3.5: Compute scene flow = P1 - P0
            Vec3f scene_vec(P1.x - P0.x, P1.y - P0.y, P1.z - P0.z); // 计算场景流向量
            // 将计算得到的场景流向量存储到输出图像中
            sceneFlow.at<Vec3f>(v, u) = scene_vec;

            valid_points++; // 统计有效点的数量
        }
    }

    cout << "Scene flow computation completed." << endl;
    cout << "Valid points: " << valid_points << "/" << total_points
         << " (" << (100.0f * valid_points / total_points) << "%)" << endl;

    return sceneFlow;
}

// Helper function: reproject 2D pixel + disparity to 3D point
Point3f SceneFlowReconstructor::reprojectTo3D(int u, int v, float disparity) const
{
    if (disparity <= 0.0f)
    {
        return Point3f(0, 0, 0); // Invalid point
    }

    // 使用传入的相机参数替代硬编码的KITTI参数
    /**
     * \brief 使用相机内参将像素坐标和视差重投影到3D空间
     * \param u 像素的水平坐标
     * \param v 像素的垂直坐标
     * \param disparity 像素的视差值
     * \return 返回重投影后的3D点 (X, Y, Z)
     * \details 使用相机内参 fx, fy, cx, cy 和基线 baseline 进行重投影。
     * 计算公式为：
     * Z = (fx * baseline) / disparity;
     * X = (u - cx) * Z / fx;
     * Y = (v - cy) * Z / fy;
     * 这里的 fx, fy, cx, cy, baseline 是在构造函数中设置的。
     * 如果视差为0或负值，返回 (0, 0, 0) 表示无效点。
     */
    // 计算深度 Z（前向方向）
    float Z = (fx_ * baseline_) / disparity;
    float X = (u - cx_) * Z / fx_;
    float Y = (v - cy_) * Z / fy_;

    return Point3f(X, Y, Z);
}

// Helper function: validate disparity value
bool SceneFlowReconstructor::isValidDisparity(float disparity) const
{
    /**
     * disparity >= MIN_DISPARITY	排除过小视差（如为 0，表示无匹配）
     * disparity <= MAX_DISPARITY	排除不合理的大视差
     * !isnan(...)	不是 NaN（Not a Number）
     * !isinf(...)	不是正无穷或负无穷
     */
    return (disparity >= MIN_DISPARITY && disparity <= MAX_DISPARITY &&
            !isnan(disparity) && !isinf(disparity));
}

// Helper function: validate 3D point
bool SceneFlowReconstructor::isValidPoint(const cv::Point3f &point) const
{
    /**
     * point.z > 0.1f	排除过小的深度（如 0，表示无效点）
     * point.z < MAX_DEPTH	排除过大的深度（如 100 米，表示超出场景范围）
     * !isnan(...)	不是 NaN（Not a Number）
     * !isinf(...)	不是正无穷或负无穷
     */
    return (point.z > 0.1f && point.z < MAX_DEPTH &&
            !isnan(point.x) && !isnan(point.y) && !isnan(point.z) &&
            !isinf(point.x) && !isinf(point.y) && !isinf(point.z));
}