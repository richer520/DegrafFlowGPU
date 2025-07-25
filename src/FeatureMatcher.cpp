/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow
\author Felix Stephenson
*/

#include "stdafx.h"
#include "FeatureMatcher.h"
#include <cuda_runtime.h>  // ✅ 确保包含CUDA运行时API
#include <chrono>          // ✅ 确保包含时间测量


// 保存匹配点为 InterpoNet 输入格式
void save_matches(const std::vector<cv::Point2f>& src_pts, const std::vector<cv::Point2f>& dst_pts, const std::string& filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < src_pts.size(); ++i) {
        file << src_pts[i].x << " " << src_pts[i].y << " "
             << dst_pts[i].x << " " << dst_pts[i].y << std::endl;
    }
    file.close();
}

// 保存边缘图为 .dat 文件
void save_edge_dat(const cv::Mat& edge_map, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(edge_map.data), edge_map.total());
    ofs.close();
}

// ✅ 静态成员初始化
CudaGradientDetector* FeatureMatcher::shared_gpu_detector = nullptr;
std::mutex FeatureMatcher::gpu_detector_mutex;
bool FeatureMatcher::gpu_detector_initialized = false;

cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> FeatureMatcher::shared_cuda_tracker = nullptr;
std::mutex FeatureMatcher::cuda_tracker_mutex;
bool FeatureMatcher::cuda_tracker_initialized = false;
// Constructor
FeatureMatcher::FeatureMatcher()
{
}

// ✅ 新增：GPU检测器单例获取方法
CudaGradientDetector* FeatureMatcher::getGPUDetector() {
    std::lock_guard<std::mutex> lock(gpu_detector_mutex);
    
    if (!gpu_detector_initialized) {
        std::cout << "🚀 Initializing shared GPU detector (one-time setup)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        shared_gpu_detector = new CudaGradientDetector();
        gpu_detector_initialized = true;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "✅ Shared GPU detector initialized in " << duration.count() << " ms" << std::endl;
    }
    
    return shared_gpu_detector;
}

// ✅ 新增：CUDA SparsePyrLK单例获取方法
cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> FeatureMatcher::getCudaTracker() {
    std::lock_guard<std::mutex> lock(cuda_tracker_mutex);
    
    if (!cuda_tracker_initialized) {
        std::cout << "🚀 Initializing shared CUDA tracker (one-time setup)..." << std::endl;
        
        shared_cuda_tracker = cv::cuda::SparsePyrLKOpticalFlow::create();
        
        // 设置参数 (与原来一致)
        shared_cuda_tracker->setWinSize(cv::Size(21, 21));
        shared_cuda_tracker->setMaxLevel(4);
        shared_cuda_tracker->setNumIters(30);
        shared_cuda_tracker->setUseInitialFlow(false);
        
        cuda_tracker_initialized = true;
        std::cout << "✅ Shared CUDA tracker initialized" << std::endl;
    }
    
    return shared_cuda_tracker;
}

// ✅ 新增：CUDA SparsePyrLK预热方法
void FeatureMatcher::warmupCudaSparsePyrLK() {
    std::cout << "🔥 Warming up CUDA SparsePyrLK..." << std::endl;
    
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cout << "⚠️ No CUDA devices available for warmup" << std::endl;
            return;
        }
        
        // 创建小测试图像和特征点
        cv::Mat test_img1 = cv::Mat::ones(64, 64, CV_8UC1);
        cv::Mat test_img2 = cv::Mat::ones(64, 64, CV_8UC1);
        
        std::vector<cv::Point2f> test_points = {
            cv::Point2f(10, 10), cv::Point2f(20, 20), cv::Point2f(30, 30)
        };
        
        // 上传到GPU
        cv::cuda::GpuMat gpu_img1, gpu_img2;
        gpu_img1.upload(test_img1);
        gpu_img2.upload(test_img2);
        
        cv::Mat test_points_mat(1, test_points.size(), CV_32FC2, (void*)&test_points[0]);
        cv::cuda::GpuMat gpu_points, gpu_tracked_points, gpu_status, gpu_error;
        gpu_points.upload(test_points_mat);
        
        // 执行一次计算触发内核编译
        auto tracker = getCudaTracker();
        tracker->calc(gpu_img1, gpu_img2, gpu_points, gpu_tracked_points, gpu_status, gpu_error);
        
        // cv::cuda::deviceSynchronize();
		cudaDeviceSynchronize();
        std::cout << "✅ CUDA SparsePyrLK warmed up successfully" << std::endl;
        
    } catch (cv::Exception &e) {
        std::cout << "⚠️ CUDA SparsePyrLK warmup failed: " << e.what() << std::endl;
    }
}

// ✅ 新增：清理方法
void FeatureMatcher::cleanup() {
    std::lock_guard<std::mutex> lock1(gpu_detector_mutex);
    std::lock_guard<std::mutex> lock2(cuda_tracker_mutex);
    
    if (shared_gpu_detector) {
        delete shared_gpu_detector;
        shared_gpu_detector = nullptr;
        gpu_detector_initialized = false;
        std::cout << "🧹 Cleaned up shared GPU detector" << std::endl;
    }
    
    if (shared_cuda_tracker) {
        shared_cuda_tracker.release();
        cuda_tracker_initialized = false;
        std::cout << "🧹 Cleaned up shared CUDA tracker" << std::endl;
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

void FeatureMatcher::degraf_flow_LK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
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

	///////////////// 4. Variational refinement (optional - not used in final results as adds significant computation time) ///////////////

	// Split flow image into x and y components to pass to refinement
	// Mat U_V[2];   //destination array
	// split(dense_flow, U_V);//split source

	// int64 refinement_start = getTickCount();
	//// Source images to gray
	// Mat gray_1;
	// Mat gray_2;
	// cvtColor(from, gray_1, cv::COLOR_RGB2GRAY);
	// cvtColor(to, gray_2, cv::COLOR_RGB2GRAY);

	// int variational_refinement_iter = 3;
	// float variational_refinement_alpha = 20.f;
	// float variational_refinement_gamma = 10.f;
	// float variational_refinement_delta = 5.f;

	// Ptr<cv::optflow::VariationalRefinement> variational_refinement_processor = cv::optflow::createVariationalFlowRefinement();

	// variational_refinement_processor->setAlpha(variational_refinement_alpha);
	// variational_refinement_processor->setDelta(variational_refinement_delta);
	// variational_refinement_processor->setGamma(variational_refinement_gamma);
	// variational_refinement_processor->setSorIterations(5);
	// variational_refinement_processor->setFixedPointIterations(variational_refinement_iter);

	// variational_refinement_processor->calcUV(gray_1, gray_2, U_V[0], U_V[1]);

	// variational_refinement_processor->collectGarbage();

	// vector<Mat> ch;
	// flow.create(from.size(), CV_32FC2);
	// Mat dst = flow.getMat();
	// ch.push_back(U_V[0]);
	// ch.push_back(U_V[1]);
	// merge(ch, dst);
}

// DeGraF-Flow using Robust Local Optical Flow point tracking, requires RLOF code found at https://github.com/tsenst/RLOFLib
/*!
\param from first image
\param to second image, same size and type as from
\param flow h output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/
void FeatureMatcher::degraf_flow_RLOF(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
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
			int gpu_status = gpu_gradient_detector->DetectGradients(saliency_mat, 3, 3, 9, 9); // Same DeGraF params
			
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			
			if (gpu_status) {
				// Convert GPU keypoints to Point2f (same interface as CPU version)
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
				
				std::cout << "GPU DeGraF detection completed successfully!" << std::endl;
				std::cout << "GPU Detection time: " << gpu_duration.count() / 1000.0 << " ms" << std::endl;
				std::cout << "GPU Keypoints detected: " << points.size() << std::endl;
				
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




// DeGraF-Flow using CUDA SparsePyrLK Optical Flow point tracking
/*!
\param from first image
\param to second image, same size and type as from
\param flow output optical flow, 2 channel image (middlebury format)
\param k number of support vectors used by the interpolator
\param sigma, use_post_proc, fgs_lambda, fgs_sigma EdgeAwareInterpolator params defined in openCV documentation
*/
void FeatureMatcher::degraf_flow_CudaLK(InputArray from, InputArray to, OutputArray flow, int k, float sigma, bool use_post_proc, float fgs_lambda, float fgs_sigma)
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
	vector<float> err;
	int64 timeStart0 = getTickCount();
	
	// Compare different feature point inputs DeGraF, FAST, SIFT, SURF, AGAST, ORB, Grid.
	int point = 7;
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
			std::cout << "Final points for CudaLK: " << points.size() << std::endl;
		}
		// Clean up IplImage headers
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
	}
	// Using other point detectors (保持不变)
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
		Ptr<SIFT> detector = SIFT::create(5400, 3, 0.00, 100.0);
		detector->detect(prev, keypoints);
		cv::KeyPoint::convert(keypoints, points);
	}
	else if (point == 7) {
		// GPU-accelerated DeGraF detection (保持完全不变)
		cv::Size s = from.size();
		
		Mat fromMat = from.getMat();
		IplImage *fromIpl = cvCreateImageHeader(cvSize(fromMat.cols, fromMat.rows), IPL_DEPTH_8U, fromMat.channels());
		cvSetData(fromIpl, fromMat.data, fromMat.step);
		Mat dogMat = Mat::zeros(s.height, s.width, CV_8UC3);
		IplImage *dogIpl = cvCreateImageHeader(cvSize(dogMat.cols, dogMat.rows), IPL_DEPTH_8U, dogMat.channels());
		cvSetData(dogIpl, dogMat.data, dogMat.step);
		
		std::cout << "About to call GPU DeGraF saliency detection" << std::endl;
		
		SaliencyDetector saliency_detector;
		saliency_detector.DoGoS_Saliency(fromIpl, dogIpl, 3, true, true);
		saliency_detector.Release();
		
		cv::Mat saliency_mat = cv::cvarrToMat(dogIpl);
		
		std::cout << "Starting GPU DeGraF gradient detection..." << std::endl;
		auto gpu_start = std::chrono::high_resolution_clock::now();
		
		// CudaGradientDetector *gpu_gradient_detector = new CudaGradientDetector();
		// ✅ 关键修改：使用单例GPU检测器
        CudaGradientDetector* gpu_gradient_detector = getGPUDetector();
		try {
			int gpu_status = gpu_gradient_detector->DetectGradients(saliency_mat, 3, 3, 9, 9);
			
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			
			if (gpu_status) {
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
				
				std::cout << "GPU DeGraF detection completed successfully!" << std::endl;
				std::cout << "GPU Detection time: " << gpu_duration.count() / 1000.0 << " ms" << std::endl;
				std::cout << "GPU Keypoints detected: " << points.size() << std::endl;

				const int MAX_POINTS = 30000;
				if (points.size() > MAX_POINTS) {
					std::cout << "Subsampling " << points.size() << " points to " << MAX_POINTS 
							<< " for EPIC compatibility" << std::endl;
					
					std::vector<cv::Point2f> subsampled_points;
					subsampled_points.reserve(MAX_POINTS);
					
					int step_sample = points.size() / MAX_POINTS;
					for (size_t i = 0; i < points.size(); i += step_sample) {
						subsampled_points.push_back(points[i]);
						if (subsampled_points.size() >= MAX_POINTS) break;
					}
					
					points = subsampled_points;
					std::cout << "Final points for CudaLK: " << points.size() << std::endl;
				}
			}
		} catch (const std::exception& e) {
			std::cerr << "GPU DeGraF exception: " << e.what() << std::endl;
			std::cerr << "Falling back to CPU version..." << std::endl;
			
			GradientDetector *cpu_gradient_detector = new GradientDetector();
			int cpu_status = cpu_gradient_detector->DetectGradients(dogIpl, 3, 3, 9, 9);
			
			if (cpu_status) {
				cv::KeyPoint::convert(cpu_gradient_detector->keypoints, points);
				std::cout << "CPU fallback successful, keypoints: " << points.size() << std::endl;
			}
			
			delete cpu_gradient_detector;
		}
		
		// delete gpu_gradient_detector;
		cvReleaseImageHeader(&fromIpl);
		cvReleaseImageHeader(&dogIpl);
		
		std::cout << "GPU DeGraF detection phase completed" << std::endl;
	}

	long double execTime0 = (getTickCount() * 1.0000 - timeStart0) / (getTickFrequency() * 1.0000);
	std::cout << "Time to compute DeGraF points = " << execTime0 << "\n\n";

	//////////////////////////////// CUDA SparsePyrLK ////////////////////////////////////////////////////////////////

	int64 timeStart1 = getTickCount();

	// 准备数据
	std::vector<Point2f> prevPoints, currPoints;
	for (int r = 0; r < points.size(); r++)
	{
		prevPoints.push_back(Point2f(points[r].x, points[r].y));
	}

	// 预分配输出容器
	currPoints.resize(prevPoints.size());
	std::vector<uchar> status(prevPoints.size());
	std::vector<float> error(prevPoints.size());

	std::cout << "points size: " << points.size() << std::endl;
	std::cout << "prevPoints size: " << prevPoints.size() << std::endl;
	std::cout << "currPoints size before calc: " << currPoints.size() << std::endl;
	std::cout << "Input images - prev channels: " << prev_grayscale.channels() 
			<< ", cur channels: " << cur_grayscale.channels() << std::endl;

	try
	{
		// 检查 CUDA 设备支持
		if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
			throw cv::Exception(-1, "No CUDA capable devices found", __FUNCTION__, __FILE__, __LINE__);
		}

		// 上传图像到 GPU
		cv::cuda::GpuMat gpu_prev, gpu_cur;
		gpu_prev.upload(prev_grayscale);
		gpu_cur.upload(cur_grayscale);

		// 上传特征点到 GPU
		cv::cuda::GpuMat gpu_prevPoints, gpu_currPoints;
		cv::cuda::GpuMat gpu_status, gpu_error;
		
		// 将 vector<Point2f> 转换为正确格式的 Mat
		if (!prevPoints.empty()) {
			cv::Mat prevPointsMat(1, prevPoints.size(), CV_32FC2, (void*)&prevPoints[0]);
			gpu_prevPoints.upload(prevPointsMat);
			std::cout << "Uploaded prevPoints: rows=" << prevPointsMat.rows 
					<< ", cols=" << prevPointsMat.cols 
					<< ", type=" << prevPointsMat.type() << std::endl;
		}

		// 创建 CUDA SparsePyrLK 追踪器
		// cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> cuda_tracker = 
		// 	cv::cuda::SparsePyrLKOpticalFlow::create();

		// ✅ 关键修改：使用单例CUDA追踪器
        auto cuda_tracker = getCudaTracker();
		// 设置参数 (对应 RLOF 的参数)
		cuda_tracker->setWinSize(cv::Size(21, 21));  // 对应 RLOF 的窗口大小
		cuda_tracker->setMaxLevel(4);                // 对应 maxLevel
		cuda_tracker->setNumIters(30);              // 对应 maxIteration
		cuda_tracker->setUseInitialFlow(false);     // 不使用初始光流

		std::cout << "Starting CUDA SparsePyrLK optical flow..." << std::endl;

		// 执行 GPU 光流追踪
		cuda_tracker->calc(gpu_prev, gpu_cur, gpu_prevPoints, gpu_currPoints, gpu_status, gpu_error);

		// 下载结果回 CPU
		cv::Mat currPointsMat, statusMat, errorMat;
		gpu_currPoints.download(currPointsMat);
		gpu_status.download(statusMat);
		gpu_error.download(errorMat);

		// 转换回 vector 格式
		if (!currPointsMat.empty() && currPointsMat.rows > 0) {
			currPoints.assign((Point2f*)currPointsMat.datastart, (Point2f*)currPointsMat.dataend);
		}
		if (!statusMat.empty() && statusMat.rows > 0) {
			status.assign(statusMat.datastart, statusMat.dataend);
		}
		if (!errorMat.empty() && errorMat.rows > 0) {
			error.assign((float*)errorMat.datastart, (float*)errorMat.dataend);
		}

		std::cout << "CUDA SparsePyrLK calculation completed successfully" << std::endl;
	}
	catch (cv::Exception &e)
	{
		std::cout << "CUDA SparsePyrLK Error: " << e.what() << std::endl;
		std::cout << "Falling back to CPU Lucas-Kanade..." << std::endl;
		
		// CPU fallback using standard Lucas-Kanade
		cv::calcOpticalFlowPyrLK(prev_grayscale, cur_grayscale, prevPoints, currPoints, status, error,
								cv::Size(21, 21), 4, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
		
		std::cout << "CPU Lucas-Kanade fallback completed" << std::endl;
	}

	// 清理和过滤 (完全保持原有逻辑)
	dst_points_filtered.clear();
	points_filtered.clear();
	dst_points_filtered.shrink_to_fit(); 
	points_filtered.shrink_to_fit();

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
	std::cout << "Time to run CUDA SparsePyrLK = " << execTime1 << "\n\n";
	std::cout << "Filtered points: " << points_filtered.size() << " out of " << points.size() << std::endl;

	////////////////////////////////   Interpolation (完全不变) //////////////////////////////////////////////////////////////////

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


void FeatureMatcher::degraf_flow_InterpoNet(InputArray from, InputArray to, OutputArray flow)
{
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
			int gpu_status = gpu_gradient_detector->DetectGradients(saliency_mat, 3, 3, 9, 9); // Same DeGraF params
			
			auto gpu_end = std::chrono::high_resolution_clock::now();
			auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
			
			if (gpu_status) {
				// Convert GPU keypoints to Point2f (same interface as CPU version)
				cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
				
				std::cout << "GPU DeGraF detection completed successfully!" << std::endl;
				std::cout << "GPU Detection time: " << gpu_duration.count() / 1000.0 << " ms" << std::endl;
				std::cout << "GPU Keypoints detected: " << points.size() << std::endl;
				
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

	// 保存匹配点
	save_matches(points_filtered, dst_points_filtered, "../external/interponet_tf/InterpoNet/inputs/temp_matches.txt");

	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> sed =
    cv::ximgproc::createStructuredEdgeDetection("../external/interponet_tf/InterpoNet/inputs/model.yml");

	cv::Mat imgFloat, edges;
	prev.convertTo(imgFloat, CV_32FC3, 1.0f / 255.0f);
	sed->detectEdges(imgFloat, edges);
	cv::imwrite("../external/interponet_tf/InterpoNet/inputs/test_edges.png", edges * 255);


	save_edge_dat(edges, "../external/interponet_tf/InterpoNet/inputs/temp_edges.dat");


	////////////////////////////////   Interpolation (完全不变) //////////////////////////////////////////////////////////////////

	// int64 timeStart2 = getTickCount();

	// if (points_filtered.size() > SHRT_MAX)
	// {
	// 	cout << "Too many points to interpolate";
	// }

	// flow.create(from.size(), CV_32FC2);
	// Mat dense_flow = flow.getMat();

	// Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
	// gd->setK(k);
	// gd->setSigma(sigma);
	// gd->setUsePostProcessing(use_post_proc);
	// gd->setFGSLambda(fgs_lambda);
	// gd->setFGSSigma(fgs_sigma);

	// gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);

	// long double execTime2 = (getTickCount() * 1.0000 - timeStart2) / (getTickFrequency() * 1.0000);
	// std::cout << "Time to interpolate = " << execTime2 << "\n";
}




// 🔧 辅助函数：创建稀疏光流图
Mat FeatureMatcher::createSparseFlowMap(const std::vector<Point2f>& src_points, 
                                       const std::vector<Point2f>& dst_points, 
                                       Size image_size) {
	Mat sparse_flow = Mat::zeros(image_size, CV_32FC2);
	
	for (size_t i = 0; i < src_points.size() && i < dst_points.size(); i++) {
		Point2f flow_vec = dst_points[i] - src_points[i];
		
		// 确保坐标在图像范围内
		int x = (int)round(src_points[i].x);
		int y = (int)round(src_points[i].y);
		
		if (x >= 0 && x < image_size.width && y >= 0 && y < image_size.height) {
			sparse_flow.at<Vec2f>(y, x) = Vec2f(flow_vec.x, flow_vec.y);
		}
	}
	
	return sparse_flow;
}