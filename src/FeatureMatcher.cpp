/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow and InterpoNet Optical flow
\author Gang Wang
*/

#include "stdafx.h"
#include "FeatureMatcher.h"
#include <chrono>
#include "utils.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <json/json.h>

#if __has_include(<opencv2/xfeatures2d/nonfree.hpp>)
#include <opencv2/xfeatures2d/nonfree.hpp>
#define DEGRAF_HAVE_XFEATURES2D 1
#else
#define DEGRAF_HAVE_XFEATURES2D 0
#endif

#if USE_CUDA
#include <cuda_runtime.h>
#include "degraf_detector.h"
#endif

namespace {
inline cv::Ptr<cv::Feature2D> createSiftDetector(int nfeatures, int nOctaveLayers,
                                                  double contrastThreshold, double edgeThreshold,
                                                  double sigma) {
#if DEGRAF_HAVE_XFEATURES2D
	return cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
	return cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#endif
}
}

// Constructor
FeatureMatcher::FeatureMatcher()
{
}

// DeGraF-Flow using lucas-kanade point tracking
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/

void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma, String num_str)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));

	int64 timeStart0 = getTickCount();
	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	vector<Point2f> dst_points;
	vector<unsigned char> status;
	vector<float> err;

	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0)
	{
		cv::Size s = from.size();

		// Convert cv::Mat to IplImage for legacy SaliencyDetector
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 5, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();

		int status_1 = gradient_detector_1->DetectGradients(dogIpl, 3, 3, 7, 7);

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);

		// Clean up IplImage headers
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
	}
	// Using other point detectors
	else if (point == 1)
	{
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints, 2, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(0, 2, 0.01, 10.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(7000, 3, 0.00, 100.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4)
	{
		vector<KeyPoint> keypoints;
		Ptr<ORB> detector = ORB::create(5400, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 5)
	{
		vector<KeyPoint> keypoints;
		Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(5, true, cv::AgastFeatureDetector::OAST_9_16);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 6)
	{

		for (int i = 0; i < prev.rows - 7; i = i + 7)
		{
			for (int j = 0; j < prev.cols - 7; j = j + 7)
			{
				points.push_back(Point2f(j, i));
			}
		}
	}

	// Lucas-Kanade point tracking
	cv::calcOpticalFlowPyrLK(prev_grayscale, cur_grayscale, points, dst_points, status, err, Size(11, 11), 4);

	// Set max vector length allowed (in pixels) N.B change max vector length for different data sets.
	int max_flow_length = 100;

	// long double execTime1 = (getTickCount()*1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	for (unsigned int i = 0; i < points.size(); i++)
	{
		if (status[i] != 0 &&
			sqrt(pow(points[i].x - dst_points[i].x, 2) + pow(points[i].y - dst_points[i].y, 2)) < max_flow_length &&
			dst_points[i].x >= 0 && dst_points[i].x < cur.cols && dst_points[i].y < cur.rows && dst_points[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(dst_points[i]);
		}
	}

	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();

	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k);
	gd->setSigma(sigma);
	gd->setUsePostProcessing(use_post_proc);
	gd->setFGSLambda(fgs_lambda);
	gd->setFGSSigma(fgs_sigma);

	if (points_filtered.size() > SHRT_MAX)
	{
		cout << "Too many points to interpolate";
	}

	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);
}

// DeGraF-Flow using Robust Local Optical Flow point tracking, requires RLOF code found at https://github.com/tsenst/RLOFLib
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/
void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma, String num_str)
{
	CV_Assert(k > 3 && sigma > 0.0001f && fgs_lambda > 1.0f && fgs_sigma > 0.01f);
	CV_Assert(!from.empty() && from.depth() == CV_8U && (from.channels() == 3 || from.channels() == 1));
	CV_Assert(!to.empty() && to.depth() == CV_8U && (to.channels() == 3 || to.channels() == 1));

	Mat prev = from.getMat();
	Mat cur = to.getMat();
	Mat prev_grayscale, cur_grayscale;

	if (prev.channels() == 3)
	{
		cvtColor(prev, prev_grayscale, COLOR_BGR2GRAY);
		cvtColor(cur, cur_grayscale, COLOR_BGR2GRAY);
	}
	else
	{
		prev.copyTo(prev_grayscale);
		cur.copyTo(cur_grayscale);
	}

	vector<Point2f> points;
	vector<Point2f> points_intermediate;
	// vector<unsigned char> status;
	vector<float> err;
	int64 timeStart0 = getTickCount();
	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 0;
	if (point == 0)
	{
		cv::Size s = from.size();

		// Convert cv::Mat to IplImage for legacy SaliencyDetector
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);
		std::cout << "About to call cvSetData for image_8u" << std::endl;
		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();

		GradientDetector *gradient_detector_1 = new GradientDetector();
		int status_1 = gradient_detector_1->DetectGradients(dogIpl, 3, 3, 9, 9); // DeGraF params specified here

		// Convert from keyPoint type to Point2f
		cv::KeyPoint::convert(gradient_detector_1->keypoints, points);

		std::cout << "CPU Detection completed successfully!" << std::endl;
		std::cout << "CPU Keypoints detected: " << points.size() << std::endl;

		// Limit the number of feature points to avoid EPIC interpolator overflow
		const int MAX_POINTS = 50000;
		if (points.size() > MAX_POINTS)
		{
			std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS
					  << " for EPIC compatibility" << std::endl;

			// Uniform subsampling strategy
			std::vector<cv::Point2f> subsampled_points;
			subsampled_points.reserve(MAX_POINTS);

			int step_sample = points.size() / MAX_POINTS;
			for (size_t i = 0; i < points.size(); i += step_sample)
			{
				subsampled_points.push_back(points[i]);
				if (subsampled_points.size() >= MAX_POINTS)
					break;
			}

			points = subsampled_points;
			std::cout << "Final points for RLOF: " << points.size() << std::endl;
		}
		// Clean up IplImage headers
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
	}
	// Using other point detectors
	else if (point == 1)
	{
		vector<KeyPoint> keypoints;
		FAST(prev, keypoints, 10, true);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 2)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(0, 2, 0.01, 10.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3)
	{
		vector<KeyPoint> keypoints;
		Ptr<ORB> detector = ORB::create(5400, 1.2f, 8, 20, 0, 2, cv::ORB::HARRIS_SCORE, 20, 20);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 4)
	{
		vector<KeyPoint> keypoints;
		Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(16, true, cv::AgastFeatureDetector::OAST_9_16);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}

	else if (point == 5)
	{
		for (int i = 0; i < prev.rows - 6; i = i + 6)
		{
			for (int j = 0; j < prev.cols - 6; j = j + 6)
			{
				points.push_back(Point2f(j, i));
			}
		}
	}
	else if (point == 6)
	{
		vector<KeyPoint> keypoints;
		Ptr<cv::Feature2D> detector = createSiftDetector(5400, 3, 0.00, 100.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 7)
	{
		// GPU-accelerated DeGraF detection
		cv::Size s = from.size();

		// Convert cv::Mat to IplImage for legacy SaliencyDetector (same as CPU version)
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);

		std::cout << "About to call GPU DeGraF saliency detection" << std::endl;

		// Saliency detection (same as CPU version)
		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();

		// Convert IplImage back to cv::Mat for detector
		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);

#if USE_CUDA
		std::cout << "Starting GPU DeGraF gradient detection..." << std::endl;
		auto gpu_start = std::chrono::high_resolution_clock::now();
		CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		try
		{
			int gpu_status = gpu_gradient_detector->CudaDetectGradients(saliency_mat, 3, 3, 9, 9); // Same DeGraF params
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			if (gpu_status)
			{
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
			}
		}
		catch (const std::exception &e)
		{
			std::cerr << "GPU DeGraF exception: " << e.what() << std::endl;
			std::cerr << "Falling back to CPU version..." << std::endl;
			GradientDetector *cpu_gradient_detector = new GradientDetector();
			int cpu_status = cpu_gradient_detector->DetectGradients(dogIpl, 3, 3, 9, 9);
			if (cpu_status)
			{
				cv::KeyPoint::convert(cpu_gradient_detector->keypoints, points);
				std::cout << "CPU fallback successful, keypoints: " << points.size() << std::endl;
			}
			delete cpu_gradient_detector;
		}
		delete gpu_gradient_detector;
#else
		GradientDetector cpu_gradient_detector;
		cpu_gradient_detector.DetectGradients(dogIpl, 3, 3, 9, 9);
		cv::KeyPoint::convert(cpu_gradient_detector.keypoints, points);
		cpu_gradient_detector.Release();
#endif

		const int MAX_POINTS = 50000;
		if (points.size() > MAX_POINTS)
		{
			std::vector<cv::Point2f> subsampled_points;
			subsampled_points.reserve(MAX_POINTS);
			int step_sample = points.size() / MAX_POINTS;
			for (size_t i = 0; i < points.size(); i += step_sample)
			{
				subsampled_points.push_back(points[i]);
				if (subsampled_points.size() >= MAX_POINTS)
					break;
			}
			points = subsampled_points;
			std::cout << "Final points for RLOF: " << points.size() << std::endl;
		}

		// Clean up IplImage headers (same as CPU version)
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);

		std::cout << "GPU DeGraF detection phase completed" << std::endl;
	}

	long double execTime0 = (getTickCount() * 1.0000 - timeStart0) / (getTickFrequency() * 1.0000);
	std::cout << "Time to compute DeGraF points = " << execTime0 << "\n\n";

	//////////////////////////////// RLOF ////////////////////////////////////////////////////////////////

	int64 timeStart1 = getTickCount();

	Mat img0 = prev;
	Mat img1 = cur;

	std::vector<Point2f> prevPoints, currPoints;
	for (int r = 0; r < points.size(); r++)
	{
		prevPoints.push_back(Point2f(points[r].x, points[r].y));
	}

	currPoints.resize(prevPoints.size());
	std::vector<uchar> status(prevPoints.size());
	std::vector<float> error(prevPoints.size());

	// Wrap the parameter structure with a pointer
	cv::Ptr<cv::optflow::RLOFOpticalFlowParameter> rlof_param =
		cv::makePtr<cv::optflow::RLOFOpticalFlowParameter>();

	rlof_param->useIlluminationModel = true;
	rlof_param->useGlobalMotionPrior = true;
	rlof_param->smallWinSize = 10;
	rlof_param->largeWinSize = 11;
	rlof_param->maxLevel = 4;
	rlof_param->maxIteration = 30;
	rlof_param->supportRegionType = cv::optflow::SR_FIXED;
	// Create RLOF optical flow estimator
	cv::Ptr<cv::optflow::SparseRLOFOpticalFlow> proc =
		cv::optflow::SparseRLOFOpticalFlow::create(rlof_param);

	std::cout << "points size: " << points.size() << std::endl;
	std::cout << "prevPoints size: " << prevPoints.size() << std::endl;
	std::cout << "currPoints size before calc: " << currPoints.size() << std::endl;
	// Check the number of input image channels
	std::cout << "Input images - prev channels: " << prev_grayscale.channels()
			  << ", cur channels: " << cur_grayscale.channels() << std::endl;
	try
	{
		proc->calc(prev_grayscale, cur_grayscale, prevPoints, currPoints, status, error);
		std::cout << "RLOF calculation completed successfully" << std::endl;
	}
	catch (cv::Exception &e)
	{
		std::cout << "OpenCV RLOF Error: " << e.what() << std::endl;
		return;
	}

	dst_points_filtered.clear();
	points_filtered.clear();
	dst_points_filtered.shrink_to_fit();
	points_filtered.shrink_to_fit();

	int max_flow_length = 100;
	for (unsigned int i = 0; i < points.size() && i < currPoints.size(); i++)
	{
		if (status[i] &&
			sqrt(pow(points[i].x - currPoints[i].x, 2) + pow(points[i].y - currPoints[i].y, 2)) < max_flow_length &&
			currPoints[i].x >= 0 && currPoints[i].x < cur.cols && currPoints[i].y < cur.rows && currPoints[i].y >= 0 &&
			points[i].x >= 0 && points[i].x < prev.cols && points[i].y < prev.rows && points[i].y >= 0)
		{
			points_filtered.push_back(points[i]);
			dst_points_filtered.push_back(currPoints[i]);
		}
	}

	long double execTime1 = (getTickCount() * 1.0000 - timeStart1) / (getTickFrequency() * 1.0000);
	std::cout << "Time to run RLOF = " << execTime1 << "\n\n";
	std::cout << "Filtered points: " << points_filtered.size() << " out of " << points.size() << std::endl;
	////////////////////////////////   Interpolation  //////////////////////////////////////////////////////////////////

	int64 timeStart2 = getTickCount();

	if (points_filtered.size() > SHRT_MAX)
	{
		cout << "Too many points to interpolate";
	}

	flow.create(from.size(), CV_32FC2);
	Mat dense_flow = flow.getMat();

	Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	gd->setK(k);
	gd->setSigma(sigma);
	gd->setUsePostProcessing(use_post_proc);
	gd->setFGSLambda(fgs_lambda);
	gd->setFGSSigma(fgs_sigma);

	gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);

	long double execTime2 = (getTickCount() * 1.0000 - timeStart2) / (getTickFrequency() * 1.0000);
	std::cout << "Time to interpolate = " << execTime2 << "\n";
}

// =====================================================
// Batch InterpoNet function (supports feature point export)
// =====================================================

std::vector<cv::Mat> FeatureMatcher::degraf_flow_InterpoNet(
	const std::vector<cv::Mat> &batch_i1,
	const std::vector<cv::Mat> &batch_i2,
	const std::vector<std::string> &batch_num_strs,
	std::vector<std::vector<cv::Point2f>> *out_points_filtered,
	std::vector<std::vector<cv::Point2f>> *out_dst_points_filtered)
{
	auto total_batch_start = std::chrono::high_resolution_clock::now();

	std::vector<cv::Mat> batch_flows;
	batch_flows.reserve(batch_i1.size());

	// Pre-create all necessary folders
	std::string raft_base_path = "../external/RAFT/data/degraf_input/";
	std::string raft_images_folder = raft_base_path + "degraf_images/";
	std::string raft_points_folder = raft_base_path + "degraf_points/";
	std::string base_path_external = "../external/InterpoNet/data/interponet_input/";

	cv::utils::fs::createDirectories(raft_images_folder);
	cv::utils::fs::createDirectories(raft_points_folder);
	cv::utils::fs::createDirectories(base_path_external);

	// =====================================================
	// Step 1: Intelligent feature detection (with caching)
	// =====================================================
	auto degraf_start = std::chrono::high_resolution_clock::now();

	std::vector<std::vector<cv::Point2f>> batch_points;
	batch_points.reserve(batch_i1.size());

	int cache_hits = 0;
	int computed = 0;

	// Save All Images
	std::vector<std::string> temp_img_paths;
	for (size_t idx = 0; idx < batch_i1.size(); ++idx)
	{
		const std::string &num_str = batch_num_strs[idx];
		std::string img_path = raft_images_folder + num_str + "_10.png";

		if (!cv::utils::fs::exists(img_path))
		{
			cv::imwrite(img_path, batch_i1[idx]);
		}
		temp_img_paths.push_back(img_path);
	}

	// Check cache and process feature detection
	for (size_t idx = 0; idx < batch_i1.size(); ++idx)
	{
		const std::string &num_str = batch_num_strs[idx];
		std::string points_path = raft_points_folder + num_str + "_points.txt";
		std::string img_path = temp_img_paths[idx];

		if (false && FlowUtils::isPointsCacheValid(points_path, img_path))
		{
			std::vector<cv::Point2f> cached_points = FlowUtils::loadCachedPoints(points_path);
			if (!cached_points.empty())
			{
				batch_points.push_back(cached_points);
				cache_hits++;
				continue;
			}
		}

		// Need to recalculate
		computed++;
		const cv::Mat &prev = batch_i1[idx];

		std::vector<cv::Point2f> points;
		cv::Size s = prev.size();

		IplImage *fromIpl = cvCreateImageHeader(cvSize(prev.cols, prev.rows), IPL_DEPTH_8U, prev.channels());
		cvSetData(fromIpl, const_cast<uchar *>(prev.data), prev.step);
		cv::Mat dogMat = cv::Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);

		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();

		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);
#if USE_CUDA
		CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		auto gpu_start = std::chrono::high_resolution_clock::now();
		gpu_gradient_detector->CudaDetectGradients(saliency_mat, 3, 3, 9, 9);
		auto gpu_end = std::chrono::high_resolution_clock::now();
		auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
		cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
		std::cout << "CUDA keypoints count: " << points.size() << std::endl;
#else
		GradientDetector cpu_gradient_detector;
		cpu_gradient_detector.DetectGradients(dogIpl, 3, 3, 9, 9);
		cv::KeyPoint::convert(cpu_gradient_detector.keypoints, points);
		cpu_gradient_detector.Release();
		std::cout << "CPU keypoints count: " << points.size() << std::endl;
#endif
		const int MAX_POINTS = 50000;
		if (points.size() > MAX_POINTS)
		{
			std::vector<cv::Point2f> subsampled_points;
			subsampled_points.reserve(MAX_POINTS);
			int step_sample = points.size() / MAX_POINTS;
			for (size_t i = 0; i < points.size(); i += step_sample)
			{
				subsampled_points.push_back(points[i]);
				if (subsampled_points.size() >= MAX_POINTS)
					break;
			}
			points = subsampled_points;
		}

#if USE_CUDA
		delete gpu_gradient_detector;
#endif
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);

		FlowUtils::savePointsToFile(points, points_path);
		batch_points.push_back(points);
	}

	auto degraf_end = std::chrono::high_resolution_clock::now();

	// =====================================================
	// Step 2: Smart Image Preparation
	// =====================================================
	auto prep_start = std::chrono::high_resolution_clock::now();

	std::vector<std::string> raft_img1_paths, raft_img2_paths, raft_points_paths, raft_matches_paths;
	std::vector<std::string> local_matches_paths;

	for (size_t idx = 0; idx < batch_i1.size(); ++idx)
	{
		const std::string &num_str = batch_num_strs[idx];

		std::string img1_path = raft_images_folder + num_str + "_10.png";
		std::string img2_path = raft_images_folder + num_str + "_11.png";

		if (!cv::utils::fs::exists(img1_path))
		{
			cv::imwrite(img1_path, batch_i1[idx]);
		}
		if (!cv::utils::fs::exists(img2_path))
		{
			cv::imwrite(img2_path, batch_i2[idx]);
		}

		raft_img1_paths.push_back("/app/data/degraf_input/degraf_images/" + num_str + "_10.png");
		raft_img2_paths.push_back("/app/data/degraf_input/degraf_images/" + num_str + "_11.png");
		raft_points_paths.push_back("/app/data/degraf_input/degraf_points/" + num_str + "_points.txt");
		raft_matches_paths.push_back("/app/data/raft_matches/" + num_str + "_matches.txt");

		// Save the local path for subsequent reading
		local_matches_paths.push_back("../external/RAFT/data/raft_matches/" + num_str + "_matches.txt");
	}

	auto prep_end = std::chrono::high_resolution_clock::now();

	// =====================================================
	// Step 3: Batch RAFT TCP calls
	// =====================================================
	auto raft_start = std::chrono::high_resolution_clock::now();

	bool raft_success = FlowUtils::callRAFTTCP_batch(raft_img1_paths, raft_img2_paths,
													 raft_points_paths, raft_matches_paths);

	auto raft_end = std::chrono::high_resolution_clock::now();

	if (raft_success)
	{
		// Directly read the first matches file for debugging
		std::string first_matches_path = local_matches_paths[0];
		std::vector<cv::Point2f> src_pts, dst_pts;
		FlowUtils::parseMatchesFile(first_matches_path, src_pts, dst_pts);
	}

	if (!raft_success)
	{
		return batch_flows;
	}

	// =====================================================
	// Step 3.5: Parse the matches file to obtain feature point pairs
	// =====================================================
	if (out_points_filtered && out_dst_points_filtered)
	{
		out_points_filtered->clear();
		out_dst_points_filtered->clear();
		out_points_filtered->reserve(batch_i1.size());
		out_dst_points_filtered->reserve(batch_i1.size());

		for (const std::string &matches_path : local_matches_paths)
		{
			std::vector<cv::Point2f> src_pts, dst_pts;
			FlowUtils::parseMatchesFile(matches_path, src_pts, dst_pts);
			out_points_filtered->push_back(src_pts);
			out_dst_points_filtered->push_back(dst_pts);
		}
	}

	// =====================================================
	// Step 4: Intelligent edge detection
	// =====================================================
	auto edge_start = std::chrono::high_resolution_clock::now();

	std::vector<std::string> interponet_img1_paths, interponet_img2_paths;
	std::vector<std::string> interponet_edges_paths, interponet_matches_paths, interponet_output_paths;

	int edge_cache_hits = 0;
	int edge_computed = 0;

	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> sed = nullptr;

	for (size_t idx = 0; idx < batch_i1.size(); ++idx)
	{
		const std::string &num_str = batch_num_strs[idx];
		const cv::Mat &prev = batch_i1[idx];

		std::string edges_dat_path = base_path_external + num_str + "_edges.dat";
		std::string img_path = raft_images_folder + num_str + "_10.png";

		if (FlowUtils::isEdgeCacheValid(edges_dat_path, img_path))
		{
			edge_cache_hits++;
		}
		else
		{
			edge_computed++;

			if (!sed)
			{
				sed = cv::ximgproc::createStructuredEdgeDetection("../external/InterpoNet/model.yml");
			}

			cv::Mat imgFloat, edges;
			prev.convertTo(imgFloat, CV_32FC3, 1.0f / 255.0f);
			sed->detectEdges(imgFloat, edges);
			FlowUtils::save_edge_dat(edges, edges_dat_path);
		}

		interponet_img1_paths.push_back("/app/external/RAFT/data/degraf_input/degraf_images/" + num_str + "_10.png");
		interponet_img2_paths.push_back("/app/external/RAFT/data/degraf_input/degraf_images/" + num_str + "_11.png");
		interponet_edges_paths.push_back("/app/external/InterpoNet/data/interponet_input/" + num_str + "_edges.dat");
		interponet_matches_paths.push_back("/app/external/RAFT/data/raft_matches/" + num_str + "_matches.txt");
		interponet_output_paths.push_back("/app/external/InterpoNet/data/interponet_output/" + num_str + "_output.flo");
	}

	auto edge_end = std::chrono::high_resolution_clock::now();

	// =====================================================
	// Step 5: Batch InterpoNet TCP calls
	// =====================================================
	auto interpo_start = std::chrono::high_resolution_clock::now();

	bool interpo_success = FlowUtils::callInterpoNetTCP_batch(
		interponet_img1_paths, interponet_img2_paths,
		interponet_edges_paths, interponet_matches_paths,
		interponet_output_paths);

	auto interpo_end = std::chrono::high_resolution_clock::now();

	if (!interpo_success)
	{
		return batch_flows;
	}

	// =====================================================
	// Step 6: Batch read results
	// =====================================================
	auto read_start = std::chrono::high_resolution_clock::now();

	for (size_t idx = 0; idx < batch_num_strs.size(); ++idx)
	{
		const std::string &num_str = batch_num_strs[idx];
		std::string flo_path = "../external/InterpoNet/data/interponet_output/" + num_str + "_output.flo";

		cv::Mat dense_flow = FlowUtils::readOpticalFlowFile(flo_path);

		if (dense_flow.empty())
		{
			dense_flow = cv::Mat::zeros(batch_i1[idx].size(), CV_32FC2);
		}

		batch_flows.push_back(dense_flow);
	}

	auto read_end = std::chrono::high_resolution_clock::now();

	return batch_flows;
}
