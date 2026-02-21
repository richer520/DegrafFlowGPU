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
This parameter constructor already sets the camera's internal parameters by default. You can call it directly; the default parameters passed to it are set in the external function.
fx, fy: The camera's focal length (in pixels), and the scale factor in the X/Y directions.
cx, cy: The position of the camera's principal point (center point) in the image.
baseline: The binocular camera's baseline (in meters), the horizontal distance between the left and right cameras.
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
// Default constructor. No camera intrinsics are set. Call setCameraParameters() to set them before use.
// The default parameter here is 0, indicating no parameters are set.
// Call setCameraParameters() to set the camera intrinsics before use.
SceneFlowReconstructor::SceneFlowReconstructor()
    : fx_(0), fy_(0), cx_(0), cy_(0), baseline_(0), camera_params_set_(false)
{
    cout << "SceneFlowReconstructor initialized. Please call setCameraParameters() before use." << endl;
}

// Set the camera internal parameter function
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
 * \brief Calculates 3D scene flow
 * \param flow 2D optical flow image, type CV_32FC2
 * \param disp0 First disparity map, type CV_32F or CV_16UC1
 * \param disp1 Second disparity map (optional), type CV_32F or CV_16UC1
 * \return Returns the calculated 3D scene flow, type CV_32FC3, where each pixel is a 3D vector
 */
Mat SceneFlowReconstructor::computeSceneFlow(const Mat &flow,
                                             const Mat &disp0,
                                             const Mat &disp1)
{
    // Step 0: Check if camera parameters are set
    if (!camera_params_set_)
    {
        cerr << "Error: Camera parameters not set! Please call setCameraParameters() first." << endl;
        return Mat();
    }

    cout << "Flow size: " << flow.cols << "x" << flow.rows << endl;
    cout << "Disp0 size: " << disp0.cols << "x" << disp0.rows << endl;
    cout << "Flow type: " << flow.type() << endl;
    cout << "Disp0 type: " << disp0.type() << endl;

    // Input validation.
    CV_Assert(!flow.empty() && flow.type() == CV_32FC2);
    CV_Assert(!disp0.empty());
    CV_Assert(flow.size() == disp0.size());

    cout << "Computing scene flow..." << endl;
    cout << "Flow dimensions: " << flow.cols << "x" << flow.rows << endl;
    cout << "Disparity dimensions: " << disp0.cols << "x" << disp0.rows << endl;

    // Step 1: Convert disparity maps to proper format
    Mat disp0_f32, disp1_f32;

    if (disp0.type() == CV_32F)
    {
        disp0_f32 = disp0.clone();
        cout << "Using pre-converted disparity0 as float32" << endl;
    }
    else if (disp0.type() == CV_16UC1)
    {
        cout << "Warning: Received raw KITTI format, converting..." << endl;
        disp0.convertTo(disp0_f32, CV_32F, 1.0 / 256.0);
    }
    else
    {
        cerr << "Error: Unsupported disparity format: " << disp0.type() << endl;
        return Mat();
    }

    bool use_temporal_disparity = !disp1.empty();
    if (use_temporal_disparity)
    {
        CV_Assert(disp1.size() == flow.size());
        if (disp1.type() == CV_16UC1)
        {
            disp1.convertTo(disp1_f32, CV_32F, 1.0 / 256.0);
        }
        else
        {
            disp1_f32 = disp1.clone();
        }
        cout << "Using temporal disparity information." << endl;
    }
    else
    {
        cout << "Using single disparity map (constant depth assumption)." << endl;
    }

    // Step 2: Initialize scene flow output (3D motion vectors)
    Mat sceneFlow(flow.size(), CV_32FC3, Scalar::all(0));

    int valid_points = 0;
    int total_points = flow.rows * flow.cols;

    // Step 3: Process each pixel
    for (int v = 0; v < flow.rows; ++v)
    {
        for (int u = 0; u < flow.cols; ++u)
        {
            // Step 3.0: Get the optical flow vector
            Point2f flow_vec = flow.at<Point2f>(v, u);
            if (!std::isfinite(flow_vec.x) || !std::isfinite(flow_vec.y))
            {
                continue;
            }
            float disp_t0 = disp0_f32.at<float>(v, u);

            // Skip invalid disparity at time t
            if (!isValidDisparity(disp_t0))
            {
                continue;
            }

            // Step 3.1: Compute 3D position at time t
            Point3f P0 = reprojectTo3D(u, v, disp_t0);
            if (!isValidPoint(P0))
            {
                continue;
            }

            // Step 3.2: Calculate target pixel position using optical flow
            float u1 = static_cast<float>(u) + flow_vec.x;
            float v1 = static_cast<float>(v) + flow_vec.y;

            // Check bounds for target position
            if (u1 < 0.0f || u1 > static_cast<float>(flow.cols - 1) ||
                v1 < 0.0f || v1 > static_cast<float>(flow.rows - 1))
            {
                continue;
            }

            // Step 3.3: Get disparity at time t+1
            float disp_t1;
            if (use_temporal_disparity)
            {
                if (!sampleDisparityBilinear(disp1_f32, u1, v1, disp_t1))
                {
                    continue;
                }
            }
            else
            {
                // Constant depth assumption: disp_t1 = disp_t0
                disp_t1 = disp_t0;
            }

            if (!isValidDisparity(disp_t1))
            {
                continue;
            }

            // Step 3.4: Compute 3D position at time t+1
            Point3f P1 = reprojectTo3D(u1, v1, disp_t1);
            if (!isValidPoint(P1))
            {
                continue;
            }

            // Step 3.5: Compute scene flow = P1 - P0
            Vec3f scene_vec(P1.x - P0.x, P1.y - P0.y, P1.z - P0.z);
            sceneFlow.at<Vec3f>(v, u) = scene_vec;

            valid_points++; // Count the number of valid points
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
    return reprojectTo3D(static_cast<float>(u), static_cast<float>(v), disparity);
}

Point3f SceneFlowReconstructor::reprojectTo3D(float u, float v, float disparity) const
{
    if (disparity <= 0.0f)
    {
        return Point3f(0, 0, 0); // Invalid point
    }
    float Z = (fx_ * baseline_) / disparity;
    float X = (u - cx_) * Z / fx_;
    float Y = (v - cy_) * Z / fy_;

    return Point3f(X, Y, Z);
}

bool SceneFlowReconstructor::sampleDisparityBilinear(const cv::Mat &disp, float u, float v, float &value) const
{
    if (disp.empty() || disp.type() != CV_32F) return false;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    if (u < 0.0f || v < 0.0f || u > static_cast<float>(disp.cols - 1) || v > static_cast<float>(disp.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const int x1 = std::min(x0 + 1, disp.cols - 1);
    const int y1 = std::min(y0 + 1, disp.rows - 1);

    const float dx = u - static_cast<float>(x0);
    const float dy = v - static_cast<float>(y0);

    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w10 = dx * (1.0f - dy);
    const float w01 = (1.0f - dx) * dy;
    const float w11 = dx * dy;

    const float d00 = disp.at<float>(y0, x0);
    const float d10 = disp.at<float>(y0, x1);
    const float d01 = disp.at<float>(y1, x0);
    const float d11 = disp.at<float>(y1, x1);

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;

    if (isValidDisparity(d00)) { weighted_sum += w00 * d00; weight_sum += w00; }
    if (isValidDisparity(d10)) { weighted_sum += w10 * d10; weight_sum += w10; }
    if (isValidDisparity(d01)) { weighted_sum += w01 * d01; weight_sum += w01; }
    if (isValidDisparity(d11)) { weighted_sum += w11 * d11; weight_sum += w11; }

    if (weight_sum <= 1e-6f) return false;
    value = weighted_sum / weight_sum;
    return true;
}

// Helper function: validate disparity value
bool SceneFlowReconstructor::isValidDisparity(float disparity) const
{
    return (disparity >= MIN_DISPARITY && disparity <= MAX_DISPARITY &&
            !isnan(disparity) && !isinf(disparity));
}

// Helper function: validate 3D point
bool SceneFlowReconstructor::isValidPoint(const cv::Point3f &point) const
{
    return (point.z > 0.1f && point.z < MAX_DEPTH &&
            !isnan(point.x) && !isnan(point.y) && !isnan(point.z) &&
            !isinf(point.x) && !isinf(point.y) && !isinf(point.z));
}