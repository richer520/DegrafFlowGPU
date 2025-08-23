// Degraf_2.cpp : Defines the entry point for the console application
// Author: Felix Stephenson
// N.B  Must first specify image file locations in the run_evaluation function in EvaluateOptFlow class

#include "stdafx.h"

// FS code
#include "FeatureMatcher.h"
#include "SaliencyDetector.h"
#include "EvaluateOptFlow.h"
#include "EvaluateSceneFlow.h"
#include "vo_features.h"

// OpenCV - requires contrib modules
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{

	////////////////////////// Flow evaluation //////////////////////////
	// *** Must first specify image file locations in the run_evaluation function in EvaluateOptFlow class ***

	// 配置参数
	bool use_batch_processing = true;  // 是否使用批量处理模式
	int batch_size = 5;               // 批量处理大小

	// 多方法评估配置
	std::vector<String> methods = {
		"degraf_flow_interponet",   // GPU加速的InterpoNet方法
		// "degraf_flow_rlof",         // Baseline RLOF方法
		// "deepflow",              // deepflow方法
	};

	bool display_images = false;        // 是否显示可视化
	int start_image = 0;                // 起始图像编号
	int total_images = 10;              // 总评估帧数

	// 实例化评估器
	EvaluateOptFlow evaluator;

	// 存储每个方法的结果
	std::map<String, std::vector<OptFlowMetrics>> method_results;

	// 对每个方法进行评估
	for (const auto& method : methods) {
		cout << "\n--- Evaluating optical flow method: " << method << " ---" << endl;
		
		// 清空之前的结果
		evaluator.clearResults();
		
		if (use_batch_processing && method == "degraf_flow_interponet") {
			// InterpoNet使用批量处理
			cout << "Using batch processing for " << method << " (batch_size=" << batch_size << ")" << endl;
			
			for (int batch_idx = 0; batch_idx < total_images; batch_idx += batch_size) {
				int current_batch_size = std::min(batch_size, total_images - batch_idx);
				std::vector<int> batch_indices;
				for (int i = 0; i < current_batch_size; ++i) {
					batch_indices.push_back(start_image + batch_idx + i);
				}
				
				cout << "Processing batch: frames " << batch_indices.front() 
					<< "-" << batch_indices.back() << endl;
				
				// 调用统一接口的批量版本
				std::vector<OptFlowMetrics> batch_results = 
					evaluator.runEvaluation(method, display_images, batch_indices);
				
				// 收集结果
				for (const auto& result : batch_results) {
					method_results[method].push_back(result);
				}
			}
		} else {
			// 其他方法逐帧处理
			for (int i = start_image; i < start_image + total_images; ++i) {
				cout << "\nOPTICAL FLOW [" << method << "] Frame #: " << i << "\n";
				
				// 调用统一接口的单帧版本
				std::vector<int> single_index = {i};
				std::vector<OptFlowMetrics> single_result = 
					evaluator.runEvaluation(method, display_images, single_index);
				
				if (!single_result.empty()) {
					method_results[method].push_back(single_result[0]);
				}
			}
		}
		
		// 输出该方法的统计结果 - 保持原格式完全一致
		const auto& results = method_results[method];
		if (!results.empty()) {
			cout << "\n--- Optical Flow Stats [" << method << "] ---\n";
			cout << "# EPE      STD      0.5     1        2        3       5       10       time\n";
			
			double avg_EPE = 0, avg_std = 0, avg_R05 = 0, avg_R1 = 0, avg_R2 = 0, avg_R3 = 0, avg_R5 = 0, avg_R10 = 0, avg_time = 0;
			
			// 显示每帧详细数据
			for (size_t j = 0; j < results.size(); j++) {
				const OptFlowMetrics &metrics = results[j];
				
				printf("%d %.4f %.4f %.2f %.2f %.2f %.2f %.2f %.2f %.3f\n",
					metrics.image_no, metrics.EPE, metrics.std_dev,
					metrics.R05, metrics.R1, metrics.R2, metrics.R3,
					metrics.R5, metrics.R10, metrics.time_ms);
				
				avg_EPE += metrics.EPE;
				avg_std += metrics.std_dev;
				avg_R05 += metrics.R05;
				avg_R1 += metrics.R1;
				avg_R2 += metrics.R2;
				avg_R3 += metrics.R3;
				avg_R5 += metrics.R5;
				avg_R10 += metrics.R10;
				avg_time += metrics.time_ms;
			}
			
			// 输出平均值 - 保持原格式完全一致
			size_t count = results.size();
			cout << "\nAverages for " << method << ":\n";
			cout << "Average EPE: " << avg_EPE / count << "\n";
			cout << "Average R2.0: " << avg_R2 / count << "\n";
			cout << "Average R3.0: " << avg_R3 / count << "\n";
			cout << "Average Time: " << avg_time / count << " ms\n";
			cout << "Average STD: " << avg_std / count << "\n";
			cout << "--------------------------------------------\n";
		}
	}

	////////////////////////// 3D Scene Flow evaluation //////////////////////////
	
	// 配置参数
	// bool use_batch_processing = true;  // 是否使用批量处理模式
	// int batch_size = 5;               // 批量处理大小
	
	// // 多方法评估配置
	// std::vector<std::string> methods = {
	// 	"degraf_flow_interponet",   // GPU加速的InterpoNet方法
	// 	// "degraf_flow_rlof",         // Baseline RLOF方法
	// 	// "deepflow",           // deepflow方法
	// };
	
	// bool display_images = false;        // 是否显示可视化
	// int start_image = 0;                // 起始图像编号
	// int total_images = 10;              // 总评估帧数

	// // 实例化评估器
	// EvaluateSceneFlow evaluator;

	// // 存储每个方法的结果
	// std::map<std::string, std::vector<SceneFlowMetrics>> method_results;
	
	// // 对每个方法进行评估
	// for (const auto& method : methods) {
	// 	cout << "\n--- Evaluating method: " << method << " ---" << endl;
		
	// 	// 清空之前的结果
	// 	evaluator.clearResults();
		
	// 	if (use_batch_processing && method == "degraf_flow_interponet") {
	// 		// InterpoNet使用批量处理
	// 		cout << "Using batch processing for " << method << " (batch_size=" << batch_size << ")" << endl;
			
	// 		for (int batch_idx = 0; batch_idx < total_images; batch_idx += batch_size) {
	// 			int current_batch_size = std::min(batch_size, total_images - batch_idx);
	// 			std::vector<int> batch_indices;
	// 			for (int i = 0; i < current_batch_size; ++i) {
	// 				batch_indices.push_back(start_image + batch_idx + i);
	// 			}
				
	// 			cout << "Processing batch: frames " << batch_indices.front() 
	// 				 << "-" << batch_indices.back() << endl;
				
	// 			// 调用统一接口的批量版本
	// 			std::vector<SceneFlowMetrics> batch_results = 
	// 				evaluator.runEvaluation(method, display_images, batch_indices);
				
	// 			// 收集结果
	// 			method_results[method].insert(
	// 				method_results[method].end(), 
	// 				batch_results.begin(), 
	// 				batch_results.end()
	// 			);
	// 		}
	// 	} else {
	// 		// 其他方法逐帧处理
	// 		for (int i = start_image; i < start_image + total_images; ++i) {
	// 			cout << "\nSCENE FLOW [" << method << "] Frame #: " << i << "\n";
				
	// 			// 调用统一接口的单帧版本
	// 			SceneFlowMetrics result = evaluator.runEvaluation(method, display_images, i);
	// 			method_results[method].push_back(result);
	// 		}
	// 	}
		
	// 	// 输出该方法的统计结果 - 保持原格式完全一致
	// 	const auto& results = method_results[method];
	// 	if (!results.empty()) {
	// 		cout << "\n--- Scene Flow Stats [" << method << "] ---\n";
	// 		cout << "# EPE3d(m)    AccS(%)    AccR(%)    Outlier(%)    Valid    Time(ms)\n";
			
	// 		double avg_EPE3d = 0, avg_AccS = 0, avg_AccR = 0, avg_Outlier = 0, avg_time = 0;
			
	// 		// 显示每帧详细数据
	// 		for (size_t j = 0; j < results.size(); j++) {
	// 			const SceneFlowMetrics &metrics = results[j];
				
	// 			printf("  %.4f      %.2f       %.2f       %.2f         %d      %.1f\n",
	// 				   metrics.EPE3d, metrics.AccS, metrics.AccR,
	// 				   metrics.Outlier, metrics.valid_count, metrics.time_ms);
				
	// 			avg_EPE3d += metrics.EPE3d;
	// 			avg_AccS += metrics.AccS;
	// 			avg_AccR += metrics.AccR;
	// 			avg_Outlier += metrics.Outlier;
	// 			avg_time += metrics.time_ms;
	// 		}
			
	// 		// 输出平均值 - 保持原格式完全一致
	// 		size_t count = results.size();
	// 		cout << "\nAverages for " << method << ":\n";
	// 		cout << "Average EPE3d: " << avg_EPE3d / count << " m\n";
	// 		cout << "Average AccS: " << avg_AccS / count << "%\n";
	// 		cout << "Average AccR: " << avg_AccR / count << "%\n";
	// 		cout << "Average Outlier: " << avg_Outlier / count << "%\n";
	// 		cout << "Average Time: " << avg_time / count << " ms\n";
	// 	}
	// }
	// // 导出对比表格
	// evaluator.exportSceneFlowComparisonCSV("../data/outputs/scene_flow_comparison.csv", method_results);
	// cout << "Scene flow comparison table exported to: ../data/outputs/scene_flow_comparison.csv\n";
	// cout << "================================================\n";

	/////////////////////////////////////////////////////////////////////////

	/////////////////    Odometry   ////////////////////////
	// Ensure all VO data file locations are specified in the Odometry class
	// Odometry vo = Odometry();
	// vo.run();
	///////////////////////////////////////////////////////

	//////////////// SAMPLE VIDEO ///////////////////
	// Writes a demo video of degraf flow on a KITTI odometry image sequence

	// VideoWriter outputVideo;
	//
	// Mat t = imread("D:/data_odometry_gray/dataset/sequences/00/image_0/000000.png"); // change this
	// outputVideo.open("C:/Users/felix/OneDrive/Documents/Uni/Year 4/project/Images/output/Videos/trialflow_3.avi", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 10, Size(1241,752), true);

	// if (!outputVideo.isOpened())
	//{
	//	cout << "Could not open the output video for write: " << endl;
	//	return -1;
	// }

	// FeatureMatcher f = FeatureMatcher();

	// int MAX_FRAME = 500;
	// char filename1[200];
	// char filename2[200];
	// sprintf(filename1, "D:/data_odometry_gray/dataset/sequences/00/image_0/%06d.png", 0); // change this to dir with list of video frames
	// sprintf(filename2, "D:/data_odometry_gray/dataset/sequences/00/image_0/%06d.png", 1);
	// Mat img_1 = imread(filename1);
	// Mat img_2 = imread(filename2);

	// Mat prevImage = img_2;
	// Mat currImage;
	// Mat flow;

	// char filename[100];

	// for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
	//	cout << "\nFRAME #: " << numFrame << "\n";
	//	sprintf(filename, "D:/data_odometry_gray/dataset/sequences/00/image_0/%06d.png", numFrame);
	//	currImage = imread(filename);

	//	f.degraf_flow_RLOF(prevImage, currImage, flow, 8, 127, (0.05000000075F), true, (500.0F), (1.5F));

	//	cv::Mat win_mat(cv::Size(1241, 752), CV_8UC3);

	//	flow = flowToDisplay(flow)*1.5;
	//	flow.convertTo(flow, CV_8UC3, 255.0);
	//	cout << t.size();
	//	cout << "\ntypes:  " << flow.type() << ", " << prevImage.type();

	//	// Copy small images into big mat
	//	flow.copyTo(win_mat(cv::Rect(0, 376, 1241, 376)));
	//	prevImage.copyTo(win_mat(cv::Rect(0, 0, 1241, 376)));

	//	outputVideo << win_mat;

	//	imshow("prev", win_mat);
	//	//imshow("curr", currImage);
	//	//imshow("flow", flow);
	//	waitKey(1);
	//	prevImage = currImage.clone();
	//}
	// break;

	return 0;
}