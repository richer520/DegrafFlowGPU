
#include "stdafx.h"

// FS code
#include "FeatureMatcher.h"
#include "SaliencyDetector.h"
#include "EvaluateOptFlow.h"
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
	////////////////////////// Optical Flow (CPU-only) //////////////////////////
	// *** Must first specify image file locations in the run_evaluation function in EvaluateOptFlow class ***

	// Configuration parameters
	bool use_batch_processing = false;  // CPU-only path: run frame by frame
	int batch_size = 5;               // Batch size

	// Multi-method evaluation configuration
	std::vector<String> methods = {
		"degraf_flow_lk",           // CPU DeGraF + LK
		"degraf_flow_rlof",         // Baseline RLOF
		// "deepflow",           // deepflow
		// "farneback", // farneback
		// "tvl1", 	// tvl1
		// "DISflow_fast", // DISflow_fast
		// "DISflow_medium", // DISflow_medium
	};

	bool display_images = false;        
	int start_image = 0;                
	int total_images = 10;              

	EvaluateOptFlow evaluator;
	std::map<String, std::vector<OptFlowMetrics>> method_results;
	cout << "Running CPU-only optical flow evaluation (LK + RLOF)." << endl;
	cout << "Scene flow evaluation is disabled in this mode." << endl;

	for (const auto& method : methods) {
		cout << "\n--- Evaluating optical flow method: " << method << " ---" << endl;
		
		evaluator.clearResults();
		
		if (use_batch_processing && method == "degraf_flow_interponet") {
			// InterpoNet uses batch processing
			cout << "Using batch processing for " << method << " (batch_size=" << batch_size << ")" << endl;
			
			for (int batch_idx = 0; batch_idx < total_images; batch_idx += batch_size) {
				int current_batch_size = std::min(batch_size, total_images - batch_idx);
				std::vector<int> batch_indices;
				for (int i = 0; i < current_batch_size; ++i) {
					batch_indices.push_back(start_image + batch_idx + i);
				}
				
				cout << "Processing batch: frames " << batch_indices.front() 
					<< "-" << batch_indices.back() << endl;
				
				// Call the batch version of the unified interface
				std::vector<OptFlowMetrics> batch_results = 
					evaluator.runEvaluation(method, display_images, batch_indices);
				
				
				for (const auto& result : batch_results) {
					method_results[method].push_back(result);
				}
			}
		} else {
			for (int i = start_image; i < start_image + total_images; ++i) {
				cout << "\nOPTICAL FLOW [" << method << "] Frame #: " << i << "\n";
				
				// Call the single-frame version of the unified interface
				std::vector<int> single_index = {i};
				std::vector<OptFlowMetrics> single_result = 
					evaluator.runEvaluation(method, display_images, single_index);
				
				if (!single_result.empty()) {
					method_results[method].push_back(single_result[0]);
				}
			}
		}
		
		const auto& results = method_results[method];
		if (!results.empty()) {
			cout << "\n--- Optical Flow Stats [" << method << "] ---\n";
			cout << "# EPE      STD      0.5     1        2        3       5       10       time\n";
			
			double avg_EPE = 0, avg_std = 0, avg_R05 = 0, avg_R1 = 0, avg_R2 = 0, avg_R3 = 0, avg_R5 = 0, avg_R10 = 0, avg_time = 0;
			
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

	return 0;
}