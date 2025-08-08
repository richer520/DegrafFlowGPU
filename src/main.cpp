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

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{

	////////////////////////// Flow evaluation //////////////////////////
	// *** Must first specify image file locations in the run_evaluation function in EvaluateOptFlow class ***

	// EvaluateOptFlow e = EvaluateOptFlow();
	// int no_of_images = 2; // Number of image pairs to loop though

	// // Run evaluation of a given optical flow method (see EvaluateOptFlow.cpp for all available methods)
	// for (int i = 0; i < no_of_images; i++)
	// {
	// 	cout << "\nIMAGE 1113#: " << i << "\n\n";
	// 	e.runEvaluation("degraf_flow_interponet", false, i); // specify flow method here
	// }

	// // Output all stats and averages
	// cout << "---------------   Stats  -------------------\n";
	// std::cout << "Build date: " << __DATE__ << " " << __TIME__ << std::endl;
	// float EPE = 0;
	// float time = 0;
	// float std = 0;
	// float three = 0;
	// float two = 0;
	// cout << "# EPE      STD      0.5     1        2        3       5       10       time  \n";
	// for (size_t j = 0; j < no_of_images; j++)
	// {
	// 	EPE += e.all_stats[j][1];
	// 	time += e.all_stats[j][9];
	// 	std += e.all_stats[j][2];
	// 	two += e.all_stats[j][5];
	// 	three += e.all_stats[j][6];
	// 	for (size_t k = 0; k < e.all_stats[0].size(); k++)
	// 	{
	// 		cout << e.all_stats[j][k] << " ";
	// 	}
	// 	cout << "\n\n";
	// }
	// cout << "Average EPE: " << EPE / no_of_images << "\n\n";

	// cout << "Average R2.0: " << two / no_of_images << "\n\n";

	// cout << "Average R3.0: " << three / no_of_images << "\n\n";

	// cout << "Average Time: " << time / no_of_images << "\n\n";

	// cout << "Average STD: " << std / no_of_images << "\n\n";
	// cout << "--------------------------------------------" << "\n\n";

	////////////////////////// 3D Scene Flow evaluation //////////////////////////
	// 📊 场景流评估部分
	std::string method = "degraf_flow_rlof"; // 评估的光流方法
	bool display_images = false;		   // 是否显示可视化
	int start_image = 0;				   // 起始图像编号
	int no_of_images_scene = 6;			   // 要评估的帧数

	// 实例化评估器
	EvaluateSceneFlow evaluator;

	// 批量评估循环
	for (int i = start_image; i < start_image + no_of_images_scene; ++i)
	{
		cout << "\nSCENE FLOW [" << method << "] Frame #: " << i << "\n";
		SceneFlowMetrics result = evaluator.runEvaluation(method, display_images, i);
	}

	// 输出统计结果
	cout << "\n---------------   Scene Flow Stats [" << method << "]  -------------------\n";
	cout << "# EPE3d(m)    AccS(%)    AccR(%)    Outlier(%)    Valid    Time(ms)\n";

	// 计算平均值
	double avg_EPE3d = 0;
	double avg_AccS = 0;
	double avg_AccR = 0;
	double avg_Outlier = 0;
	double avg_time = 0;

	// 输出每帧详细结果并累加
	for (size_t j = 0; j < evaluator.getAllResults().size(); j++)
	{
		const SceneFlowMetrics &metrics = evaluator.getAllResults()[j];

		printf("  %.4f      %.2f       %.2f       %.2f         %d      %.1f\n",
			   metrics.EPE3d, metrics.AccS, metrics.AccR,
			   metrics.Outlier, metrics.valid_count, metrics.time_ms);

		// 累加用于计算平均值
		avg_EPE3d += metrics.EPE3d;
		avg_AccS += metrics.AccS;
		avg_AccR += metrics.AccR;
		avg_Outlier += metrics.Outlier;
		avg_time += metrics.time_ms;
	}

	// 计算并输出平均值
	size_t count = evaluator.getAllResults().size();
	if (count > 0)
	{
		cout << "\nAverages:\n";
		cout << "Average EPE3d: " << avg_EPE3d / count << " m\n";
		cout << "Average AccS: " << avg_AccS / count << "%\n";
		cout << "Average AccR: " << avg_AccR / count << "%\n";
		cout << "Average Outlier: " << avg_Outlier / count << "%\n";
		cout << "Average Time: " << avg_time / count << " ms\n";
	}

	cout << "--------------------------------------------\n";
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