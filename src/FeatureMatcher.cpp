/*!
\file FeatureMatcher.cpp
\brief Functions that compute DeGraF-Flow using both Lucas-Kanade and Robust Local Optical flow
\author Felix Stephenson
*/

#include "stdafx.h"
#include "FeatureMatcher.h"

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
		int status_1 = gradient_detector_1->DetectGradients(dogIpl, 3, 3, 3, 3); // DeGraF params specified here

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
			int gpu_status = gpu_gradient_detector->DetectGradients(saliency_mat, 3, 3, 3, 3); // Same DeGraF params
			
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