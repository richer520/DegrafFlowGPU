/*!
\file FeatureMatcher.h
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow
\author Felix Stephenson
*/

#pragma once

#include "GradientDetector.h"
#include "SaliencyDetector.h"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/types.hpp> 
#include <opencv2/features2d.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/ximgproc.hpp>
#include "opencv2/ximgproc/sparse_match_interpolator.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include "degraf_detector.h"
#include <iostream>	 // standard C++ I/O
#include <string>	 // standard C++ I/O
#include <algorithm> // includes max()
#include <vector>
#include <opencv2/core/utils/filesystem.hpp>

// N.B need RLOF code from https://github.com/tsenst/RLOFLib
// #include <RLOF_Flow.h>
#include <opencv2/optflow/rlofflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
using namespace cv;

class FeatureMatcher
{

public:
	// Public variables
	std::vector<Point2f> points_filtered, dst_points_filtered; // corresponding points in each image

	// Public functions
	FeatureMatcher();
	void degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str);
	void degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str);
	void degraf_flow_CudaLK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str);
    void degraf_flow_InterpoNet(InputArray from, InputArray to, OutputArray flow,String num_str);
	// void degraf_flow_GPU(InputArray from, InputArray to, OutputArray flow, int radius, float eps, bool use_post_proc, float fgs_lambda, float fgs_sigma);

	// ✅ 新增：单例管理方法
    static CudaGradientDetector* getGPUDetector();
    static cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> getCudaTracker();
    static void cleanup();
    static void warmupCudaSparsePyrLK();
    bool callInterpoNetTCP(const std::string& img1_path,
        const std::string& img2_path,
        const std::string& edge_path,
        const std::string& match_path,
        const std::string& out_path);
    
    bool callRAFTTCP(const std::string& image1_path,
        const std::string& image2_path,
        const std::string& points_path,
        const std::string& output_path);

private:
	// ✅ 添加：GPU检测器单例管理
    static CudaGradientDetector* shared_gpu_detector;
    static std::mutex gpu_detector_mutex;
    static bool gpu_detector_initialized;
    
    // ✅ 添加：CUDA SparsePyrLK单例管理
    static cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> shared_cuda_tracker;
    static std::mutex cuda_tracker_mutex;
    static bool cuda_tracker_initialized;
};

