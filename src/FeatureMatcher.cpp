/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow
\author Felix Stephenson
*/

#include "stdafx.h"
#include "FeatureMatcher.h"
#include <cuda_runtime.h>  // ✅ 确保包含CUDA运行时API
#include <chrono>          // ✅ 确保包含时间测量
#include <sys/socket.h>
#include <arpa/inet.h>
#include "degraf_detector.h" 
#include <unistd.h>
#include <json/json.h>  // 需要安装libjsoncpp-dev


// Constructor
FeatureMatcher::FeatureMatcher()
{
}

/**
 * @brief 从RAFT生成的matches文件解析特征点对
 * @param matches_path matches文件路径
 * @param src_points 输出源图像特征点
 * @param dst_points 输出目标图像特征点
 */
void FeatureMatcher::parseMatchesFile(const std::string& matches_path,
								std::vector<cv::Point2f>& src_points,
								std::vector<cv::Point2f>& dst_points) {
	src_points.clear();
	dst_points.clear();

	std::ifstream file(matches_path);
	if (!file.is_open()) {
		return;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		float x0, y0, x1, y1;
		if (iss >> x0 >> y0 >> x1 >> y1) {
			src_points.emplace_back(x0, y0);
			dst_points.emplace_back(x1, y1);
		}
	}
	file.close();
}


void save_edge_dat(const cv::Mat& edge_map, const std::string& filename) {
    // 1. 转成 float32
    cv::Mat edge_float;
    if (edge_map.type() != CV_32F)
        edge_map.convertTo(edge_float, CV_32F);
    else
        edge_float = edge_map;

    // 2. 强制转为单通道灰度图
    if (edge_float.channels() > 1)
        cv::cvtColor(edge_float, edge_float, cv::COLOR_BGR2GRAY);

    // 3. 检查维度
    std::cout << "[DEBUG] edge_float shape: " << edge_float.rows << "x" << edge_float.cols 
              << ", channels: " << edge_float.channels() << std::endl;

    // 4. 写入 binary 文件
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(edge_float.data),
              edge_float.rows * edge_float.cols * sizeof(float));
    ofs.close();

    std::cout << "[DEBUG] Saved edge.dat: " << edge_float.rows << "x" << edge_float.cols
              << " = " << edge_float.total() << " floats" << std::endl;
}


// Replacement for cv::optflow::readOpticalFlow - reads .flo files
static Mat readOpticalFlowFile(const String &path)
{
	std::ifstream file(path.c_str(), std::ios_base::binary);
	if (!file.good())
	{
		printf("Error opening flow file: %s\n", path.c_str());
		return Mat();
	}

	float magic;
	file.read((char *)&magic, sizeof(float));
	if (magic != 202021.25f) // .flo file magic number
	{
		printf("Invalid .flo file magic number\n");
		return Mat();
	}

	int width, height;
	file.read((char *)&width, sizeof(int));
	file.read((char *)&height, sizeof(int));

	Mat flow(height, width, CV_32FC2);
	file.read((char *)flow.data, width * height * 2 * sizeof(float));
	file.close();

	return flow;
}


// 保存特征点为RAFT格式的txt文件
void savePointsToFile(const std::vector<cv::Point2f>& points, const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        for (const auto& point : points) {
            file << point.x << " " << point.y << "\n";
        }
        file.close();
        std::cout << "[DEBUG] Saved " << points.size() << " points to: " << filepath << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to open file for writing: " << filepath << std::endl;
    }
}

// DeGraF-Flow using lucas-kanade point tracking
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/

void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str)
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
		// 在 OpenCV 4.5 中，SIFT 在主命名空间 cv 中，不是 xfeatures2d
		Ptr<SIFT> detector = SIFT::create(0, 2, 0.01, 10.0, 1.6);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 3)
	{
		vector<KeyPoint> keypoints;
		// 修复：使用 cv::SIFT 而不是 xfeatures2d::SIFT
		Ptr<SIFT> detector = SIFT::create(7000, 3, 0.00, 100.0);
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
void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma,String num_str)
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
		
		// 限制特征点数量以避免EPIC插值器溢出
		const int MAX_POINTS = 30000;  // EPIC插值器的安全上限
		if (points.size() > MAX_POINTS) {
			std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS 
					<< " for EPIC compatibility" << std::endl;
			
			// 均匀子采样策略
			std::vector<cv::Point2f> subsampled_points;
			subsampled_points.reserve(MAX_POINTS);
			
			int step_sample = points.size() / MAX_POINTS;
			for (size_t i = 0; i < points.size(); i += step_sample) {
				subsampled_points.push_back(points[i]);
				if (subsampled_points.size() >= MAX_POINTS) break;
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
		Ptr<SIFT> detector = SIFT::create(0, 2, 0.01, 10.0, 1.6);
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
		// 修复：使用 cv::SIFT 而不是 xfeatures2d::SIFT
		Ptr<SIFT> detector = SIFT::create(5400, 3, 0.00, 100.0);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 7) {
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
		
		// Convert IplImage back to cv::Mat for GPU detector
		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);
		
		std::cout << "Starting GPU DeGraF gradient detection..." << std::endl;
		auto gpu_start = std::chrono::high_resolution_clock::now();
		
		// GPU DeGraF detection
		CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		
		try {
			int gpu_status = gpu_gradient_detector->CudaDetectGradients(saliency_mat, 3, 3, 9, 9); // Same DeGraF params
			
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			
			if (gpu_status) {
				// Convert GPU keypoints to Point2f (same interface as CPU version)
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
				
				// std::cout << "GPU DeGraF detection completed successfully!" << std::endl;
				// std::cout << "GPU Detection time: " << gpu_duration
				// on.count() / 1000.0 << " ms" << std::endl;
				// std::cout << "GPU Keypoints detected: " << points.size() << std::endl;
				
				// Optional: GPU memory usage info
				// size_t total_bytes, gradient_bytes, keypoint_bytes;
				// gpu_gradient_detector->GetMemoryUsage(total_bytes, gradient_bytes, keypoint_bytes);
				// std::cout << "GPU Memory used: " << total_bytes / (1024*1024) << " MB" << std::endl;

				// 限制特征点数量以避免EPIC插值器溢出
				const int MAX_POINTS = 30000;  // EPIC插值器的安全上限
				if (points.size() > MAX_POINTS) {
					std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS 
							<< " for EPIC compatibility" << std::endl;
					
					// 均匀子采样策略
					std::vector<cv::Point2f> subsampled_points;
					subsampled_points.reserve(MAX_POINTS);
					
					int step_sample = points.size() / MAX_POINTS;
					for (size_t i = 0; i < points.size(); i += step_sample) {
						subsampled_points.push_back(points[i]);
						if (subsampled_points.size() >= MAX_POINTS) break;
					}
					
					points = subsampled_points;
					std::cout << "Final points for RLOF: " << points.size() << std::endl;
				}
				
			}
			
		} catch (const std::exception& e) {
			std::cerr << "GPU DeGraF exception: " << e.what() << std::endl;
			std::cerr << "Falling back to CPU version..." << std::endl;
			
			// Fallback to CPU version
			GradientDetector *cpu_gradient_detector = new GradientDetector();
			int cpu_status = cpu_gradient_detector->DetectGradients(dogIpl, 3, 3, 9, 9);
			
			if (cpu_status) {
				cv::KeyPoint::convert(cpu_gradient_detector->keypoints, points);
				std::cout << "CPU fallback successful, keypoints: " << points.size() << std::endl;
			}
			
			delete cpu_gradient_detector;
		}
		
		// Clean up GPU detector
		delete gpu_gradient_detector;
		
		// Clean up IplImage headers (same as CPU version)
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
		
		std::cout << "GPU DeGraF detection phase completed" << std::endl;
	}

	long double execTime0 = (getTickCount() * 1.0000 - timeStart0) / (getTickFrequency() * 1.0000);
	std::cout << "Time to compute DeGraF points = " << execTime0 << "\n\n";

	//////////////////////////////// RLOF ////////////////////////////////////////////////////////////////

	int64 timeStart1 = getTickCount();

	// 使用 Mat 替代 rlof::Image
	Mat img0 = prev;
	Mat img1 = cur;

	// 保留 prevPoints, currPoints 变量名
	std::vector<Point2f> prevPoints, currPoints;
	for (int r = 0; r < points.size(); r++)
	{
		prevPoints.push_back(Point2f(points[r].x, points[r].y));
	}

	// 预先初始化输出容器 - 关键修复！
	currPoints.resize(prevPoints.size()); // 预分配空间
	std::vector<uchar> status(prevPoints.size()); // 预分配空间
	std::vector<float> error(prevPoints.size()); // 预分配空间

	// 用指针包装参数结构体
	cv::Ptr<cv::optflow::RLOFOpticalFlowParameter> rlof_param = 
		cv::makePtr<cv::optflow::RLOFOpticalFlowParameter>();

	rlof_param->useIlluminationModel = true;
	rlof_param->useGlobalMotionPrior = true;
	rlof_param->smallWinSize = 10;
	rlof_param->largeWinSize = 11;
	rlof_param->maxLevel = 4;
	rlof_param->maxIteration = 30;
	rlof_param->supportRegionType = cv::optflow::SR_FIXED;
	// 创建 RLOF 光流估计器
	cv::Ptr<cv::optflow::SparseRLOFOpticalFlow> proc =
		cv::optflow::SparseRLOFOpticalFlow::create(rlof_param);

	std::cout << "points size: " << points.size() << std::endl;
	std::cout << "prevPoints size: " << prevPoints.size() << std::endl;
	std::cout << "currPoints size before calc: " << currPoints.size() << std::endl;
	// 检查输入图像通道数
	std::cout << "Input images - prev channels: " << prev_grayscale.channels() 
			<< ", cur channels: " << cur_grayscale.channels() << std::endl;
	try
	{
		// 修复：直接传递vector，不使用包装器，使用正确的输入图像
		proc->calc(prev_grayscale, cur_grayscale, prevPoints, currPoints, status, error);
		std::cout << "RLOF calculation completed successfully" << std::endl;
	}
	catch (cv::Exception &e)
	{
		std::cout << "OpenCV RLOF Error: " << e.what() << std::endl;
		return; // 如果RLOF失败，直接返回
	}

	// 清理原有的dst_points逻辑，直接使用currPoints
	dst_points_filtered.clear();
	points_filtered.clear();
	dst_points_filtered.shrink_to_fit(); 
	points_filtered.shrink_to_fit();

	// 修复：直接使用currPoints，不需要额外的转换步骤
	int max_flow_length = 100;
	for (unsigned int i = 0; i < points.size() && i < currPoints.size(); i++)
	{
		// 检查状态是否有效
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

	// 添加CPU匹配点调试输出
	std::ofstream cpu_matches("cpu_matches_frame" + num_str + ".txt");
	for (int i = 0; i < std::min(100, (int)points_filtered.size()); i++) {
		cpu_matches << points_filtered[i].x << " " << points_filtered[i].y << " " 
					<< dst_points_filtered[i].x << " " << dst_points_filtered[i].y << std::endl;
	}
	cpu_matches.close();
	std::cout << "[DEBUG] Saved CPU matches to cpu_matches_frame" << num_str << ".txt" << std::endl;
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
// 修改后的批量InterpoNet函数（支持特征点导出）
// =====================================================

std::vector<cv::Mat> FeatureMatcher::degraf_flow_InterpoNet(
    const std::vector<cv::Mat>& batch_i1,
    const std::vector<cv::Mat>& batch_i2,
    const std::vector<std::string>& batch_num_strs,
    std::vector<std::vector<cv::Point2f>>* out_points_filtered,    // 新增输出参数
    std::vector<std::vector<cv::Point2f>>* out_dst_points_filtered) // 新增输出参数
{
    auto total_batch_start = std::chrono::high_resolution_clock::now();
    
    std::vector<cv::Mat> batch_flows;
    batch_flows.reserve(batch_i1.size());
    
    // 预先创建所有必要的文件夹
    std::string raft_base_path = "../external/RAFT/data/degraf_input/";
    std::string raft_images_folder = raft_base_path + "degraf_images/";
    std::string raft_points_folder = raft_base_path + "degraf_points/";
    std::string base_path_external = "../external/InterpoNet/data/interponet_input/";
    
    cv::utils::fs::createDirectories(raft_images_folder);
    cv::utils::fs::createDirectories(raft_points_folder);
    cv::utils::fs::createDirectories(base_path_external);
    
    // =====================================================
    // 步骤1: 智能特征检测 (带缓存)
    // =====================================================
    auto degraf_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<cv::Point2f>> batch_points;
    batch_points.reserve(batch_i1.size());
    
    int cache_hits = 0;
    int computed = 0;
    
    // 保存所有图像
    std::vector<std::string> temp_img_paths;
    for (size_t idx = 0; idx < batch_i1.size(); ++idx) {
        const std::string& num_str = batch_num_strs[idx];
        std::string img_path = raft_images_folder + num_str + "_10.png";
        
        if (!cv::utils::fs::exists(img_path)) {
            cv::imwrite(img_path, batch_i1[idx]);
        }
        temp_img_paths.push_back(img_path);
    }
    
    // 检查缓存并处理特征检测
    for (size_t idx = 0; idx < batch_i1.size(); ++idx) {
        const std::string& num_str = batch_num_strs[idx];
        std::string points_path = raft_points_folder + num_str + "_points.txt";
        std::string img_path = temp_img_paths[idx];
        
        if (false && isPointsCacheValid(points_path, img_path)) {
            std::vector<cv::Point2f> cached_points = loadCachedPoints(points_path);
            if (!cached_points.empty()) {
                batch_points.push_back(cached_points);
                cache_hits++;
                continue;
            }
        }
        
        // 需要重新计算
        computed++;
        const cv::Mat& prev = batch_i1[idx];
        
        std::vector<cv::Point2f> points;
        cv::Size s = prev.size();
        
        IplImage *fromIpl = cvCreateImageHeader(cvSize(prev.cols, prev.rows), IPL_DEPTH_8U, prev.channels());
        cvSetData(fromIpl, const_cast<uchar*>(prev.data), prev.step);
        cv::Mat dogMat = cv::Mat::zeros(s.height, s.width, CV_8UC3);
        IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
        cvSetData(dogIpl, dogMat.data, dogMat.step);
        
        SaliencyDetector saliency_detector;
        saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
        saliency_detector.Release();
        
        CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);

		// 在这里添加计时开始
		auto gpu_start = std::chrono::high_resolution_clock::now();

		gpu_gradient_detector->CudaDetectGradients(saliency_mat, 3, 3, 9, 9);

		auto gpu_end = std::chrono::high_resolution_clock::now();
		auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
		cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
        std::cout << "CUDA keypoints count: " << points.size() << std::endl;
        const int MAX_POINTS = 50000;
        if (points.size() > MAX_POINTS) {
            std::vector<cv::Point2f> subsampled_points;
            subsampled_points.reserve(MAX_POINTS);
            int step_sample = points.size() / MAX_POINTS;
            for (size_t i = 0; i < points.size(); i += step_sample) {
                subsampled_points.push_back(points[i]);
                if (subsampled_points.size() >= MAX_POINTS) break;
            }
            points = subsampled_points;
        }
        
        delete gpu_gradient_detector;
        cvReleaseImageHeader(&fromIpl);
        cvReleaseImageHeader(&dogIpl);
        
        savePointsToFile(points, points_path);
        batch_points.push_back(points);
    }
    
    auto degraf_end = std::chrono::high_resolution_clock::now();
    
    // =====================================================
    // 步骤2: 智能图像准备
    // =====================================================
    auto prep_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> raft_img1_paths, raft_img2_paths, raft_points_paths, raft_matches_paths;
    std::vector<std::string> local_matches_paths; // 本地路径用于读取
    
    for (size_t idx = 0; idx < batch_i1.size(); ++idx) {
        const std::string& num_str = batch_num_strs[idx];
        
        std::string img1_path = raft_images_folder + num_str + "_10.png";
        std::string img2_path = raft_images_folder + num_str + "_11.png";
        
        if (!cv::utils::fs::exists(img1_path)) {
            cv::imwrite(img1_path, batch_i1[idx]);
        }
        if (!cv::utils::fs::exists(img2_path)) {
            cv::imwrite(img2_path, batch_i2[idx]);
        }
        
        raft_img1_paths.push_back("/app/data/degraf_input/degraf_images/" + num_str + "_10.png");
        raft_img2_paths.push_back("/app/data/degraf_input/degraf_images/" + num_str + "_11.png");
        raft_points_paths.push_back("/app/data/degraf_input/degraf_points/" + num_str + "_points.txt");
        raft_matches_paths.push_back("/app/data/raft_matches/" + num_str + "_matches.txt");
        
        // 保存本地路径用于后续读取
        local_matches_paths.push_back("../external/RAFT/data/raft_matches/" + num_str + "_matches.txt");
    }
    
    auto prep_end = std::chrono::high_resolution_clock::now();
    
    // =====================================================
    // 步骤3: 批量 RAFT TCP 调用
    // =====================================================
    auto raft_start = std::chrono::high_resolution_clock::now();
    
    bool raft_success = callRAFTTCP_batch(raft_img1_paths, raft_img2_paths, 
                                          raft_points_paths, raft_matches_paths);
    
    auto raft_end = std::chrono::high_resolution_clock::now();

    if (raft_success) {
		// 直接读取第一个matches文件进行调试
		std::string first_matches_path = local_matches_paths[0];
		std::vector<cv::Point2f> src_pts, dst_pts;
		parseMatchesFile(first_matches_path, src_pts, dst_pts);
		
		std::ofstream gpu_matches("gpu_matches_frame" + batch_num_strs[0] + ".txt");
		for (int i = 0; i < std::min(100, (int)src_pts.size()); i++) {
			gpu_matches << src_pts[i].x << " " << src_pts[i].y << " " 
						<< dst_pts[i].x << " " << dst_pts[i].y << std::endl;
		}
		gpu_matches.close();
		std::cout << "[DEBUG] GPU matches saved with " << src_pts.size() << " points" << std::endl;
	}
	
    if (!raft_success) {
        return batch_flows;
    }
    
    // =====================================================
    // 步骤3.5: 解析matches文件获取特征点对（新增）
    // =====================================================
    if (out_points_filtered && out_dst_points_filtered) {
        out_points_filtered->clear();
        out_dst_points_filtered->clear();
        out_points_filtered->reserve(batch_i1.size());
        out_dst_points_filtered->reserve(batch_i1.size());
        
        for (const std::string& matches_path : local_matches_paths) {
            std::vector<cv::Point2f> src_pts, dst_pts;
            parseMatchesFile(matches_path, src_pts, dst_pts);
            out_points_filtered->push_back(src_pts);
            out_dst_points_filtered->push_back(dst_pts);
        }
    }
    
    // =====================================================
    // 步骤4: 智能边缘检测
    // =====================================================
    auto edge_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> interponet_img1_paths, interponet_img2_paths;
    std::vector<std::string> interponet_edges_paths, interponet_matches_paths, interponet_output_paths;
    
    int edge_cache_hits = 0;
    int edge_computed = 0;
    
    cv::Ptr<cv::ximgproc::StructuredEdgeDetection> sed = nullptr;
    
    for (size_t idx = 0; idx < batch_i1.size(); ++idx) {
        const std::string& num_str = batch_num_strs[idx];
        const cv::Mat& prev = batch_i1[idx];
        
        std::string edges_dat_path = base_path_external + num_str + "_edges.dat";
        std::string img_path = raft_images_folder + num_str + "_10.png";
        
        if (isEdgeCacheValid(edges_dat_path, img_path)) {
            edge_cache_hits++;
        } else {
            edge_computed++;
            
            if (!sed) {
                sed = cv::ximgproc::createStructuredEdgeDetection("../external/InterpoNet/model.yml");
            }
            
            cv::Mat imgFloat, edges;
            prev.convertTo(imgFloat, CV_32FC3, 1.0f / 255.0f);
            sed->detectEdges(imgFloat, edges);
            save_edge_dat(edges, edges_dat_path);
        }
        
        interponet_img1_paths.push_back("/app/external/RAFT/data/degraf_input/degraf_images/" + num_str + "_10.png");
        interponet_img2_paths.push_back("/app/external/RAFT/data/degraf_input/degraf_images/" + num_str + "_11.png");
        interponet_edges_paths.push_back("/app/external/InterpoNet/data/interponet_input/" + num_str + "_edges.dat");
        interponet_matches_paths.push_back("/app/external/RAFT/data/raft_matches/" + num_str + "_matches.txt");
        interponet_output_paths.push_back("/app/external/InterpoNet/data/interponet_output/" + num_str + "_output.flo");
    }
    
    auto edge_end = std::chrono::high_resolution_clock::now();
    
    // =====================================================
    // 步骤5: 批量 InterpoNet TCP 调用
    // =====================================================
    auto interpo_start = std::chrono::high_resolution_clock::now();
    
    bool interpo_success = callInterpoNetTCP_batch(
        interponet_img1_paths, interponet_img2_paths,
        interponet_edges_paths, interponet_matches_paths,
        interponet_output_paths
    );
    
    auto interpo_end = std::chrono::high_resolution_clock::now();
    
    if (!interpo_success) {
        return batch_flows;
    }
    
    // =====================================================
    // 步骤6: 批量读取结果
    // =====================================================
    auto read_start = std::chrono::high_resolution_clock::now();
    
    for (size_t idx = 0; idx < batch_num_strs.size(); ++idx) {
        const std::string& num_str = batch_num_strs[idx];
        std::string flo_path = "../external/InterpoNet/data/interponet_output/" + num_str + "_output.flo";
        
        cv::Mat dense_flow = readOpticalFlowFile(flo_path);
        
        if (dense_flow.empty()) {
            dense_flow = cv::Mat::zeros(batch_i1[idx].size(), CV_32FC2);
        }
        
        batch_flows.push_back(dense_flow);
    }
    
    auto read_end = std::chrono::high_resolution_clock::now();
    
    return batch_flows;
}

// 在FeatureMatcher类中添加以下辅助函数

/**
 * @brief 检查文件是否存在且时间戳比源文件新
 */
bool FeatureMatcher::isFileUpToDate(const std::string& targetFile, const std::string& sourceFile) {
    if (!cv::utils::fs::exists(targetFile)) {
        return false;
    }
    
    // 获取文件修改时间
    struct stat targetStat, sourceStat;
    if (stat(targetFile.c_str(), &targetStat) != 0 || stat(sourceFile.c_str(), &sourceStat) != 0) {
        return false;
    }
    
    // 目标文件时间戳比源文件新才认为是最新的
    return targetStat.st_mtime >= sourceStat.st_mtime;
}

/**
 * @brief 检查特征点缓存是否有效
 */
bool FeatureMatcher::isPointsCacheValid(const std::string& pointsFile, const std::string& imageFile) {
    return isFileUpToDate(pointsFile, imageFile);
}

/**
 * @brief 检查边缘检测缓存是否有效
 */
bool FeatureMatcher::isEdgeCacheValid(const std::string& edgeFile, const std::string& imageFile) {
    return isFileUpToDate(edgeFile, imageFile);
}

/**
 * @brief 加载已缓存的特征点
 */
std::vector<cv::Point2f> FeatureMatcher::loadCachedPoints(const std::string& pointsFile) {
    std::vector<cv::Point2f> points;
    std::ifstream file(pointsFile);
    if (!file.is_open()) {
        return points;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y;
        if (iss >> x >> y) {
            points.emplace_back(x, y);
        }
    }
    file.close();
    return points;
}



bool FeatureMatcher::callRAFTTCP_batch(
    const std::vector<std::string>& batch_img1_paths,
    const std::vector<std::string>& batch_img2_paths,
    const std::vector<std::string>& batch_points_paths,
    const std::vector<std::string>& batch_output_paths)
{
    // 创建socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed for batch RAFT" << std::endl;
        return false;
    }

    // 增加发送缓冲区大小
    int send_buffer_size = 4 * 1024 * 1024;  // 4MB
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, sizeof(send_buffer_size));
    
    // 服务器地址
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(9998);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    // 连接服务器
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Batch RAFT TCP connection failed" << std::endl;
        close(sock);
        return false;
    }
    
    // 构建批量JSON请求
    Json::Value request;
    request["batch_mode"] = true;
    request["batch_size"] = (int)batch_img1_paths.size();
    
    // 使用更紧凑的数组构建方式
    Json::Value img1_array(Json::arrayValue);
    Json::Value img2_array(Json::arrayValue);
    Json::Value points_array(Json::arrayValue);
    Json::Value output_array(Json::arrayValue);
    
    for (size_t i = 0; i < batch_img1_paths.size(); ++i) {
        img1_array.append(batch_img1_paths[i]);
        img2_array.append(batch_img2_paths[i]);
        points_array.append(batch_points_paths[i]);
        output_array.append(batch_output_paths[i]);
    }
    
    request["image1_paths"] = img1_array;
    request["image2_paths"] = img2_array;
    request["points_paths"] = points_array;
    request["output_paths"] = output_array;
    
    // 使用FastWriter以减少JSON大小
    Json::FastWriter writer;
    std::string request_str = writer.write(request);
    
    // 确保以换行符结束
    if (request_str.back() != '\n') {
        request_str += '\n';
    }
    
    std::cout << "[DEBUG] Sending batch RAFT request, size: " << request_str.length() << " bytes" << std::endl;
    
    // 分块发送大数据
    size_t total_sent = 0;
    size_t chunk_size = 8192;
    
    while (total_sent < request_str.length()) {
        size_t to_send = std::min(chunk_size, request_str.length() - total_sent);
        ssize_t sent = send(sock, request_str.c_str() + total_sent, to_send, 0);
        
        if (sent < 0) {
            std::cerr << "Failed to send batch RAFT request" << std::endl;
            close(sock);
            return false;
        }
        
        total_sent += sent;
    }
    
    std::cout << "[DEBUG] Sent " << total_sent << " bytes to RAFT server" << std::endl;
    
    // 接收响应
    std::string response_str;
    char buffer[4096];
    int bytes_received;
    int total_received = 0;
    int max_receive_size = 1024 * 1024;  // 最大1MB响应
    
    while (total_received < max_receive_size) {
        bytes_received = recv(sock, buffer, sizeof(buffer), 0);
        
        if (bytes_received < 0) {
            std::cerr << "Failed to receive batch RAFT response" << std::endl;
            close(sock);
            return false;
        }
        
        if (bytes_received == 0) {
            break;  // 连接关闭
        }
        
        response_str.append(buffer, bytes_received);
        total_received += bytes_received;
        
        // 检查是否收到完整响应（以换行符结束）
        if (!response_str.empty() && response_str.back() == '\n') {
            break;
        }
    }
    
    close(sock);
    
    if (response_str.empty()) {
        std::cerr << "Empty response from batch RAFT server" << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG] Received response size: " << response_str.length() << " bytes" << std::endl;
    
    // 解析响应
    Json::Value response;
    Json::CharReaderBuilder reader_builder;
    std::istringstream response_stream(response_str);
    std::string parse_errors;
    
    if (!Json::parseFromStream(reader_builder, response_stream, &response, &parse_errors)) {
        std::cerr << "Failed to parse batch RAFT response: " << parse_errors << std::endl;
        std::cerr << "Response preview: " << response_str.substr(0, 200) << "..." << std::endl;
        return false;
    }
    
    if (response["status"].asString() == "success") {
        std::cout << "[INFO] Batch RAFT TCP: " << response["message"].asString() << std::endl;
        return true;
    } else {
        std::cerr << "[ERROR] Batch RAFT TCP: " << response["message"].asString() << std::endl;
        return false;
    }
}

/**
 * @brief 批量 InterpoNet TCP 调用
 */
bool FeatureMatcher::callInterpoNetTCP_batch(
    const std::vector<std::string>& batch_img1_paths,
    const std::vector<std::string>& batch_img2_paths,
    const std::vector<std::string>& batch_edges_paths,
    const std::vector<std::string>& batch_matches_paths,
    const std::vector<std::string>& batch_output_paths)
{
    // 创建socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed for batch InterpoNet" << std::endl;
        return false;
    }
    
    // 服务器地址
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(9999);  // InterpoNet端口
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    // 连接服务器
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Batch InterpoNet TCP connection failed" << std::endl;
        close(sock);
        return false;
    }
    
    // 构建批量JSON请求
    Json::Value request;
    request["batch_mode"] = true;
    request["batch_size"] = (int)batch_img1_paths.size();
    
    Json::Value img1_array(Json::arrayValue);
    Json::Value img2_array(Json::arrayValue);
    Json::Value edges_array(Json::arrayValue);
    Json::Value matches_array(Json::arrayValue);
    Json::Value output_array(Json::arrayValue);
    
    for (size_t i = 0; i < batch_img1_paths.size(); ++i) {
        img1_array.append(batch_img1_paths[i]);
        img2_array.append(batch_img2_paths[i]);
        edges_array.append(batch_edges_paths[i]);
        matches_array.append(batch_matches_paths[i]);
        output_array.append(batch_output_paths[i]);
    }
    
    request["img1_paths"] = img1_array;
    request["img2_paths"] = img2_array;
    request["edges_paths"] = edges_array;
    request["matches_paths"] = matches_array;
    request["output_paths"] = output_array;
    
    Json::FastWriter writer;
	std::string request_str = writer.write(request);
	if (request_str.back() != '\n') {
		request_str += '\n';
	}
    
    // 发送请求
    send(sock, request_str.c_str(), request_str.length(), 0);
    
    // 接收响应
    std::string response_str;
    char buffer[4096];
    int bytes_received;
    
    while ((bytes_received = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        response_str.append(buffer, bytes_received);
        if (!response_str.empty() && response_str.back() == '\n') break;
    }
    
    close(sock);
    
    if (bytes_received <= 0) {
        std::cerr << "Failed to receive batch InterpoNet response" << std::endl;
        return false;
    }
    
    // 解析响应
    Json::Value response;
    Json::CharReaderBuilder reader_builder;
    std::istringstream response_stream(response_str);
    std::string parse_errors;
    
    if (!Json::parseFromStream(reader_builder, response_stream, &response, &parse_errors)) {
        std::cerr << "Failed to parse batch InterpoNet response: " << parse_errors << std::endl;
        return false;
    }
    
    if (response["status"].asString() == "success") {
        std::cout << "[INFO] Batch InterpoNet TCP: " << response["message"].asString() << std::endl;
        return true;
    } else {
        std::cerr << "[ERROR] Batch InterpoNet TCP: " << response["message"].asString() << std::endl;
        return false;
    }
}