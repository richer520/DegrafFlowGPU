#include "degraf_detector.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Enhanced CUDA kernel with quality filtering and sub-pixel precision
__global__ void computeGradientsEnhancedKernel(
    const float* __restrict__ image_data,
    int image_width,
    int image_height,
    int window_width,
    int window_height,
    int step_x,
    int step_y,
    int matrix_width,
    int matrix_height,
    float magnitude_threshold,
    float contrast_threshold,
    float edge_response_threshold,
    float* __restrict__ keypoint_x,
    float* __restrict__ keypoint_y,
    float* __restrict__ keypoint_response)
{
    __shared__ float window_data[49];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    if (x >= matrix_width || y >= matrix_height) return;
    
    int idx = y * matrix_width + x;
    int window_start_x = x * step_x;
    int window_start_y = y * step_y;
    
    if (window_start_x + window_width + 1 >= image_width || 
        window_start_y + window_height + 1 >= image_height) {
        keypoint_x[idx] = -1.0f;
        keypoint_y[idx] = -1.0f;
        keypoint_response[idx] = 0.0f;
        return;
    }
    
    int window_size = window_width * window_height;
    float max_value = 0.0f;
    float min_value = 1e6f;
    
    // First pass: find max/min values and load data
    for (int i = 0; i <= window_height; i++) {
        for (int j = 0; j <= window_width; j++) {
            int img_y = window_start_y + i;
            int img_x = window_start_x + j;
            float pixel_value = image_data[img_y * image_width + img_x];
            
            if (window_size <= 49 && tid == 0) {
                window_data[i * window_width + j] = pixel_value;
            }
            
            max_value = fmaxf(max_value, pixel_value);
            min_value = fminf(min_value, pixel_value);
        }
    }
    
    // Calculate local contrast for quality assessment
    float local_contrast = max_value - min_value;
    
    // Early exit for low contrast regions
    // if (local_contrast < contrast_threshold) {
    //     keypoint_x[idx] = -1.0f;
    //     keypoint_y[idx] = -1.0f;
    //     keypoint_response[idx] = 0.0f;
    //     return;
    // }
    
    // Initialize accumulators
    float divident_high_x = 0.0f;
    float divident_high_y = 0.0f;
    float divisor_high = 0.0f;
    int counter = 0;
    
    // Gradient computation for edge response calculation
    float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;
    
    // Second pass: compute centroids and gradients
    if (window_size <= 49) {
        __syncthreads();
        for (int i = 0; i < window_height; i++) {
            for (int j = 0; j < window_width; j++) {
                float pixel_value = window_data[i * window_width + j];
                
                divident_high_x += (float)(window_start_x + j) * pixel_value;
                divident_high_y += (float)(window_start_y + i) * pixel_value;
                divisor_high += pixel_value;
                counter++;
                
                // Compute gradients for edge response (simplified)
                if (i > 0 && i < window_height-1 && j > 0 && j < window_width-1) {
                    float gx = window_data[i * window_width + (j+1)] - window_data[i * window_width + (j-1)];
                    float gy = window_data[(i+1) * window_width + j] - window_data[(i-1) * window_width + j];
                    
                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
        }
    } else {
        for (int i = 0; i < window_height; i++) {
            for (int j = 0; j < window_width; j++) {
                int img_y = window_start_y + i;
                int img_x = window_start_x + j;
                float pixel_value = image_data[img_y * image_width + img_x];
                
                divident_high_x += (float)img_x * pixel_value;
                divident_high_y += (float)img_y * pixel_value;
                divisor_high += pixel_value;
                counter++;
                
                // Compute gradients for edge response
                if (i > 0 && i < window_height-1 && j > 0 && j < window_width-1) {
                    float gx = image_data[img_y * image_width + (img_x+1)] - 
                              image_data[img_y * image_width + (img_x-1)];
                    float gy = image_data[(img_y+1) * image_width + img_x] - 
                              image_data[(img_y-1) * image_width + img_x];
                    
                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
        }
    }
    
    // Calculate edge response (Harris-like)
    // float det = Ixx * Iyy - Ixy * Ixy;
    // float trace = Ixx + Iyy;
    // float edge_response = (trace > 1e-6f) ? (trace * trace / det) : 1e6f;
    
    // Edge response filtering
    // if (edge_response > edge_response_threshold) {
    //     keypoint_x[idx] = -1.0f;
    //     keypoint_y[idx] = -1.0f;
    //     keypoint_response[idx] = 0.0f;
    //     return;
    // }
    
    // Calculate final results with sub-pixel precision
    if (divisor_high > 1e-6f) {
        float centre_x = (float)window_start_x + ((float)window_width / 2.0f);
        float centre_y = (float)window_start_y + ((float)window_height / 2.0f);
        
        float centroid_x = divident_high_x / divisor_high;
        float centroid_y = divident_high_y / divisor_high;
        
        float dx = 2.0f * (centroid_x - centre_x);
        float dy = 2.0f * (centroid_y - centre_y);
        
        float magnitude = sqrtf(dx * dx + dy * dy);
        
        // Magnitude-based quality filtering
        // if (magnitude < magnitude_threshold) {
        //     keypoint_x[idx] = -1.0f;
        //     keypoint_y[idx] = -1.0f;
        //     keypoint_response[idx] = 0.0f;
        //     return;
        // }
        
        // Sub-pixel refinement using local intensity distribution
        float sub_pixel_offset_x = 0.0f;
        float sub_pixel_offset_y = 0.0f;
        
        if (window_size <= 49) {
            // Simple sub-pixel refinement for small windows
            float weight_sum = 0.0f;
            for (int i = 0; i < window_height; i++) {
                for (int j = 0; j < window_width; j++) {
                    float pixel_value = window_data[i * window_width + j];
                    float weight = pixel_value / divisor_high;
                    
                    sub_pixel_offset_x += weight * (j - window_width/2.0f) * 0.5f;
                    sub_pixel_offset_y += weight * (i - window_height/2.0f) * 0.5f;
                    weight_sum += weight;
                }
            }
            if (weight_sum > 1e-6f) {
                sub_pixel_offset_x /= weight_sum;
                sub_pixel_offset_y /= weight_sum;
            }
        }
        
        // Store results with sub-pixel precision
        keypoint_x[idx] = centroid_x + dx + sub_pixel_offset_x;
        keypoint_y[idx] = centroid_y + dy + sub_pixel_offset_y;
        keypoint_response[idx] = magnitude * local_contrast; // Combined quality score
    } else {
        keypoint_x[idx] = -1.0f;
        keypoint_y[idx] = -1.0f;
        keypoint_response[idx] = 0.0f;
    }
}

// Optimized image conversion kernel
__global__ void convertToFloat32OptimizedKernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height,
    int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    if (channels == 1) {
        dst[idx] = (float)src[idx] + 1.0f;
    } else if (channels == 3) {
        int src_idx = idx * 3;
        float gray = 0.299f * src[src_idx] + 0.587f * src[src_idx + 1] + 0.114f * src[src_idx + 2];
        dst[idx] = gray + 1.0f;
    }
}

// Constructor with pre-allocation and warmup
CudaGradientDetector::CudaGradientDetector() 
    : d_image_data(nullptr), d_keypoint_x(nullptr), d_keypoint_y(nullptr), 
      d_keypoint_response(nullptr), init_flag(false), stream(nullptr) {
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    max_image_size = 2048 * 2048;
    max_matrix_size = (2048/3) * (2048/3);
    
    size_t image_bytes = max_image_size * sizeof(float);
    size_t keypoint_bytes = max_matrix_size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_image_data, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_x, keypoint_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_y, keypoint_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_response, keypoint_bytes));
    
    warmupGPU();
    init_flag = true;
}

void CudaGradientDetector::warmupGPU() {
    try {
        cv::Mat test_img = cv::Mat::ones(64, 64, CV_8UC1);
        cv::Size test_image_size = test_img.size();
        cv::Size test_window_size(3, 3);
        int test_step_x = 9, test_step_y = 9;
        
        cv::Size test_matrix_size;
        test_matrix_size.width = (test_image_size.width - test_window_size.width) / test_step_x;
        test_matrix_size.height = (test_image_size.height - test_window_size.height) / test_step_y;
        
        int test_current_image_size = test_image_size.width * test_image_size.height;
        
        unsigned char* d_test_input;
        size_t test_input_bytes = test_img.rows * test_img.cols;
        CUDA_CHECK(cudaMalloc(&d_test_input, test_input_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_test_input, test_img.data, test_input_bytes, 
                                  cudaMemcpyHostToDevice, stream));
        
        dim3 conv_block(256);
        dim3 conv_grid((test_current_image_size + conv_block.x - 1) / conv_block.x);
        
        convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
            d_test_input, d_image_data, 
            test_img.cols, test_img.rows, test_img.channels());
        
        dim3 grad_block(16, 16);
        dim3 grad_grid((test_matrix_size.width + grad_block.x - 1) / grad_block.x,
                       (test_matrix_size.height + grad_block.y - 1) / grad_block.y);
        
        computeGradientsEnhancedKernel<<<grad_grid, grad_block, 0, stream>>>(
            d_image_data,
            test_image_size.width, test_image_size.height,
            test_window_size.width, test_window_size.height,
            test_step_x, test_step_y,
            test_matrix_size.width, test_matrix_size.height,
            0.1f, 5.0f, 10.0f, // Default quality thresholds
            d_keypoint_x, d_keypoint_y, d_keypoint_response);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_test_input));
        
    } catch (const std::exception& e) {
        std::cerr << "GPU warmup failed: " << e.what() << std::endl;
    }
}

CudaGradientDetector::~CudaGradientDetector() {
    Release();
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

int CudaGradientDetector::CudaDetectGradients(const cv::Mat& src_image, int p_window_width, 
                                          int p_window_height, int p_step_x, int p_step_y) {
    
    if (src_image.empty()) {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return 0;
    }
    
    image_size = src_image.size();
    window_size = cv::Size(p_window_width, p_window_height);
    step_x = p_step_x;
    step_y = p_step_y;
    
    matrix_size.width = (image_size.width - window_size.width) / p_step_x;
    matrix_size.height = (image_size.height - window_size.height) / p_step_y;

    // 在这里添加调试输出
    std::cout << "CUDA matrix_size: " << matrix_size.width << "x" << matrix_size.height 
    << " = " << matrix_size.width * matrix_size.height << std::endl;
    std::cout << "CUDA image_size: " << image_size.width << "x" << image_size.height << std::endl;
    std::cout << "CUDA window_size: " << window_size.width << "x" << window_size.height << std::endl;
    std::cout << "CUDA step: " << step_x << "x" << step_y << std::endl;
    
    int current_image_size = image_size.width * image_size.height;
    int current_matrix_size = matrix_size.width * matrix_size.height;
    
    if (current_image_size > max_image_size || current_matrix_size > max_matrix_size) {
        std::cerr << "Error: Image too large for pre-allocated GPU memory!" << std::endl;
        return 0;
    }
    
    cv::Mat image_8u;
    if (src_image.channels() == 3) {
        cv::cvtColor(src_image, image_8u, cv::COLOR_BGR2GRAY);
    } else {
        image_8u = src_image.clone();
    }
    
    if (image_8u.depth() != CV_8U) {
        image_8u.convertTo(image_8u, CV_8U);
    }
    
    unsigned char* d_input_image;
    size_t input_bytes = image_8u.rows * image_8u.cols;
    CUDA_CHECK(cudaMalloc(&d_input_image, input_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_input_image, image_8u.data, input_bytes, 
                              cudaMemcpyHostToDevice, stream));
    
    dim3 conv_block(256);
    dim3 conv_grid((current_image_size + conv_block.x - 1) / conv_block.x);
    
    convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
        d_input_image, d_image_data, 
        image_8u.cols, image_8u.rows, image_8u.channels());
    
    dim3 grad_block(16, 16);
    dim3 grad_grid((matrix_size.width + grad_block.x - 1) / grad_block.x,
                   (matrix_size.height + grad_block.y - 1) / grad_block.y);
    
    // Enhanced kernel with quality filtering
    float magnitude_threshold = 0.3f;      // Minimum gradient magnitude
    float contrast_threshold = 8.0f;      // Minimum local contrast
    float edge_response_threshold = 15.0f; // Maximum edge response (lower = more corner-like)
    
    computeGradientsEnhancedKernel<<<grad_grid, grad_block, 0, stream>>>(
        d_image_data,
        image_size.width, image_size.height,
        window_size.width, window_size.height,
        step_x, step_y,
        matrix_size.width, matrix_size.height,
        magnitude_threshold, contrast_threshold, edge_response_threshold,
        d_keypoint_x, d_keypoint_y, d_keypoint_response);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    h_keypoint_x.resize(current_matrix_size);
    h_keypoint_y.resize(current_matrix_size);
    h_keypoint_response.resize(current_matrix_size);
    
    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_x.data(), d_keypoint_x, 
                              current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_y.data(), d_keypoint_y, 
                              current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_keypoint_response.data(), d_keypoint_response, 
                              current_matrix_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_input_image));
    
    keypoints.clear();
    keypoints.reserve(current_matrix_size / 4);
    
    for (int i = 0; i < current_matrix_size; i++) {
        float x = h_keypoint_x[i];
        float y = h_keypoint_y[i];
        
        // 使用kernel参数中的正确变量名
        if (!isnan(x) && !isnan(y) && !isinf(x) && !isinf(y) &&
            x >= 0 && x < image_size.width && y >= 0 && y < image_size.height) {
            keypoints.emplace_back(cv::Point2f(x, y), 
                                  (float)std::min(window_size.width, window_size.height),
                                  -1, h_keypoint_response[i]);
        } else {
            // 用窗口中心替代无效点
            int matrix_x = i % matrix_size.width;
            int matrix_y = i / matrix_size.width;
            float center_x = matrix_x * step_x + window_size.width / 2.0f;
            float center_y = matrix_y * step_y + window_size.height / 2.0f;
            keypoints.emplace_back(cv::Point2f(center_x, center_y),
                                  (float)std::min(window_size.width, window_size.height),
                                  -1, 0.0f);
        }
    }
    
    return 1;
}

void CudaGradientDetector::Release() {
    if (d_image_data) { CUDA_CHECK(cudaFree(d_image_data)); d_image_data = nullptr; }
    if (d_keypoint_x) { CUDA_CHECK(cudaFree(d_keypoint_x)); d_keypoint_x = nullptr; }
    if (d_keypoint_y) { CUDA_CHECK(cudaFree(d_keypoint_y)); d_keypoint_y = nullptr; }
    if (d_keypoint_response) { CUDA_CHECK(cudaFree(d_keypoint_response)); d_keypoint_response = nullptr; }
    
    h_keypoint_x.clear();
    h_keypoint_y.clear();
    h_keypoint_response.clear();
    keypoints.clear();
    
    init_flag = false;
}

const std::vector<cv::KeyPoint>& CudaGradientDetector::GetKeypoints() const {
    return keypoints;
}

void CudaGradientDetector::GetMemoryUsage(size_t& total_bytes, size_t& gradient_bytes, size_t& keypoint_bytes) const {
    size_t image_bytes = max_image_size * sizeof(float);
    keypoint_bytes = max_matrix_size * sizeof(float) * 3;
    gradient_bytes = 0;
    total_bytes = image_bytes + keypoint_bytes;
}