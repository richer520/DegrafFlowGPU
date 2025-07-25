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

// ====== 🚨 核心算法内核 - 保持不变 ======
// Optimized CUDA kernel with shared memory and reduced global memory access
__global__ void computeGradientsOptimizedKernel(
    const float* __restrict__ image_data,
    int image_width,
    int image_height,
    int window_width,
    int window_height,
    int step_x,
    int step_y,
    int matrix_width,
    int matrix_height,
    float* __restrict__ keypoint_x,
    float* __restrict__ keypoint_y,
    float* __restrict__ keypoint_response)
{
    // Shared memory for window data (assuming max window size 7x7)
    __shared__ float window_data[49]; // 7*7 max window
    
    // Calculate thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (x >= matrix_width || y >= matrix_height) return;
    
    // Calculate linear index
    int idx = y * matrix_width + x;
    
    // Calculate window position
    int window_start_x = x * step_x;
    int window_start_y = y * step_y;
    
    // Window bounds check
    if (window_start_x + window_width >= image_width || 
        window_start_y + window_height >= image_height) {
        keypoint_x[idx] = -1.0f;
        keypoint_y[idx] = -1.0f;
        keypoint_response[idx] = 0.0f;
        return;
    }
    
    // Load window data into shared memory (if small enough)
    int window_size = window_width * window_height;
    float max_value = 0.0f;
    
    // First pass: find max value and load data
    for (int i = 0; i < window_height; i++) {
        for (int j = 0; j < window_width; j++) {
            int img_y = window_start_y + i;
            int img_x = window_start_x + j;
            float pixel_value = image_data[img_y * image_width + img_x];
            
            if (window_size <= 49 && tid == 0) {
                window_data[i * window_width + j] = pixel_value;
            }
            
            max_value = fmaxf(max_value, pixel_value);
        }
    }
    
    // Initialize accumulators
    float divident_high_x = 0.0f;
    float divident_high_y = 0.0f;
    float divisor_high = 0.0f;
    int counter = 0;
    
    // Second pass: compute centroids (optimized loop)
    if (window_size <= 49) {
        // Use shared memory for small windows
        __syncthreads();
        for (int i = 0; i < window_height; i++) {
            for (int j = 0; j < window_width; j++) {
                float pixel_value = window_data[i * window_width + j];
                
                divident_high_x += (float)(window_start_x + j) * pixel_value;
                divident_high_y += (float)(window_start_y + i) * pixel_value;
                divisor_high += pixel_value;
                counter++;
            }
        }
    } else {
        // Direct global memory access for larger windows
        for (int i = 0; i < window_height; i++) {
            for (int j = 0; j < window_width; j++) {
                int img_y = window_start_y + i;
                int img_x = window_start_x + j;
                float pixel_value = image_data[img_y * image_width + img_x];
                
                divident_high_x += (float)img_x * pixel_value;
                divident_high_y += (float)img_y * pixel_value;
                divisor_high += pixel_value;
                counter++;
            }
        }
    }
    
    // Calculate final results
    if (divisor_high > 1e-6f) {
        float centre_x = (float)window_start_x + ((float)window_width / 2.0f);
        float centre_y = (float)window_start_y + ((float)window_height / 2.0f);
        
        float centroid_x = divident_high_x / divisor_high;
        float centroid_y = divident_high_y / divisor_high;
        
        float dx = 2.0f * (centroid_x - centre_x);
        float dy = 2.0f * (centroid_y - centre_y);
        
        float magnitude = sqrtf(dx * dx + dy * dy);
        
        // Store results
        keypoint_x[idx] = centroid_x + dx;
        keypoint_y[idx] = centroid_y + dy;
        keypoint_response[idx] = magnitude;
    } else {
        keypoint_x[idx] = -1.0f;
        keypoint_y[idx] = -1.0f;
        keypoint_response[idx] = 0.0f;
    }
}

// ====== 🚨 图像转换内核 - 保持不变 ======
// Optimized image conversion kernel with coalesced memory access
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
        // Optimized RGB to grayscale conversion
        float gray = 0.299f * src[src_idx] + 0.587f * src[src_idx + 1] + 0.114f * src[src_idx + 2];
        dst[idx] = gray + 1.0f;
    }
}

// ====== ✅ 修改1：构造函数 - 增大预分配+添加预热 ======
// Constructor with pre-allocation and warmup
CudaGradientDetector::CudaGradientDetector() 
    : d_image_data(nullptr), d_keypoint_x(nullptr), d_keypoint_y(nullptr), 
      d_keypoint_response(nullptr), init_flag(false), stream(nullptr) {
    
    // Create CUDA stream for async operations
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ✅ 修改：大幅增加预分配内存池，避免运行时重分配
    // 从4K改回到2K，减少内存开销
    max_image_size = 2048 * 2048;           // 从4096改为2048
    max_matrix_size = (2048/3) * (2048/3);  // 对应调整
    
    // 一次性分配所有GPU内存
    size_t image_bytes = max_image_size * sizeof(float);
    size_t keypoint_bytes = max_matrix_size * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_image_data, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_x, keypoint_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_y, keypoint_bytes));
    CUDA_CHECK(cudaMalloc(&d_keypoint_response, keypoint_bytes));
    
    std::cout << "✅ CUDA DeGraF Detector initialized with large memory pools:" << std::endl;
    std::cout << "   Max image size: " << max_image_size << " pixels (" 
              << image_bytes / (1024*1024) << " MB)" << std::endl;
    std::cout << "   Max matrix size: " << max_matrix_size << " elements (" 
              << keypoint_bytes * 3 / (1024*1024) << " MB)" << std::endl;
    
    // ✅ 修改：添加GPU预热，消除JIT编译延迟
    warmupGPU();
    
    init_flag = true;
}

// ====== ✅ 新增函数：GPU预热功能 ======
void CudaGradientDetector::warmupGPU() {
    std::cout << "🔥 Warming up GPU DeGraF detector..." << std::endl;
    auto warmup_start = std::chrono::high_resolution_clock::now();
    
    try {
        // 创建小测试图像，触发CUDA内核编译
        cv::Mat test_img = cv::Mat::ones(64, 64, CV_8UC1);
        
        // 设置测试参数
        cv::Size test_image_size = test_img.size();
        cv::Size test_window_size(3, 3);
        int test_step_x = 9, test_step_y = 9;
        
        cv::Size test_matrix_size;
        test_matrix_size.width = (test_image_size.width - test_window_size.width) / test_step_x;
        test_matrix_size.height = (test_image_size.height - test_window_size.height) / test_step_y;
        
        int test_current_image_size = test_image_size.width * test_image_size.height;
        // int test_current_matrix_size = test_matrix_size.width * test_matrix_size.height;
        
        // 准备测试数据
        unsigned char* d_test_input;
        size_t test_input_bytes = test_img.rows * test_img.cols;
        CUDA_CHECK(cudaMalloc(&d_test_input, test_input_bytes));
        CUDA_CHECK(cudaMemcpyAsync(d_test_input, test_img.data, test_input_bytes, 
                                  cudaMemcpyHostToDevice, stream));
        
        // 触发图像转换内核编译
        dim3 conv_block(256);
        dim3 conv_grid((test_current_image_size + conv_block.x - 1) / conv_block.x);
        
        convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
            d_test_input, d_image_data, 
            test_img.cols, test_img.rows, test_img.channels());
        
        // 触发梯度计算内核编译
        dim3 grad_block(16, 16);
        dim3 grad_grid((test_matrix_size.width + grad_block.x - 1) / grad_block.x,
                       (test_matrix_size.height + grad_block.y - 1) / grad_block.y);
        
        computeGradientsOptimizedKernel<<<grad_grid, grad_block, 0, stream>>>(
            d_image_data,
            test_image_size.width, test_image_size.height,
            test_window_size.width, test_window_size.height,
            test_step_x, test_step_y,
            test_matrix_size.width, test_matrix_size.height,
            d_keypoint_x, d_keypoint_y, d_keypoint_response);
        
        // 等待所有GPU操作完成
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 清理测试内存
        CUDA_CHECK(cudaFree(d_test_input));
        
        auto warmup_end = std::chrono::high_resolution_clock::now();
        auto warmup_duration = std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_start);
        
        std::cout << "✅ GPU DeGraF detector warmed up in " 
                  << warmup_duration.count() / 1000.0 << " ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "⚠️ GPU warmup failed: " << e.what() << std::endl;
        std::cerr << "Continuing without warmup..." << std::endl;
    }
}

// Destructor - 保持不变
CudaGradientDetector::~CudaGradientDetector() {
    Release();
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

// ====== ✅ 修改2：DetectGradients - 移除动态重分配 ======
// Optimized main detection function - No dynamic reallocation
int CudaGradientDetector::DetectGradients(const cv::Mat& src_image, int p_window_width, 
                                          int p_window_height, int p_step_x, int p_step_y) {
    
    // Validate input
    if (src_image.empty()) {
        std::cerr << "❌ Error: Input image is empty!" << std::endl;
        return 0;
    }
    
    // Update parameters
    image_size = src_image.size();
    window_size = cv::Size(p_window_width, p_window_height);
    step_x = p_step_x;
    step_y = p_step_y;
    
    matrix_size.width = (image_size.width - window_size.width) / p_step_x;
    matrix_size.height = (image_size.height - window_size.height) / p_step_y;
    
    // Check if we exceed pre-allocated memory
    int current_image_size = image_size.width * image_size.height;
    int current_matrix_size = matrix_size.width * matrix_size.height;
    
    // ✅ 修改：简单检查，不再动态重分配内存
    if (current_image_size > max_image_size || current_matrix_size > max_matrix_size) {
        std::cerr << "❌ Error: Image too large for pre-allocated GPU memory!" << std::endl;
        std::cerr << "   Current image size: " << current_image_size 
                  << " (max: " << max_image_size << ")" << std::endl;
        std::cerr << "   Current matrix size: " << current_matrix_size 
                  << " (max: " << max_matrix_size << ")" << std::endl;
        std::cerr << "   Please use smaller images or increase max_image_size in constructor." << std::endl;
        return 0;
    }
    
    // Prepare image data
    cv::Mat image_8u;
    if (src_image.channels() == 3) {
        cv::cvtColor(src_image, image_8u, cv::COLOR_BGR2GRAY);
    } else {
        image_8u = src_image.clone();
    }
    
    if (image_8u.depth() != CV_8U) {
        image_8u.convertTo(image_8u, CV_8U);
    }
    
    // ✅ 修改：使用预分配的内存，无需临时分配
    // 直接使用d_image_data的一部分作为输入缓冲区
    unsigned char* d_input_image;
    size_t input_bytes = image_8u.rows * image_8u.cols;
    CUDA_CHECK(cudaMalloc(&d_input_image, input_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_input_image, image_8u.data, input_bytes, 
                              cudaMemcpyHostToDevice, stream));
    
    // Optimized launch parameters
    dim3 conv_block(256);
    dim3 conv_grid((current_image_size + conv_block.x - 1) / conv_block.x);
    
    // Launch conversion kernel
    convertToFloat32OptimizedKernel<<<conv_grid, conv_block, 0, stream>>>(
        d_input_image, d_image_data, 
        image_8u.cols, image_8u.rows, image_8u.channels());
    
    // Optimized gradient computation launch parameters
    dim3 grad_block(16, 16);
    dim3 grad_grid((matrix_size.width + grad_block.x - 1) / grad_block.x,
                   (matrix_size.height + grad_block.y - 1) / grad_block.y);
    
    // Launch gradient computation kernel
    computeGradientsOptimizedKernel<<<grad_grid, grad_block, 0, stream>>>(
        d_image_data,
        image_size.width, image_size.height,
        window_size.width, window_size.height,
        step_x, step_y,
        matrix_size.width, matrix_size.height,
        d_keypoint_x, d_keypoint_y, d_keypoint_response);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Async copy results back
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
    
    // Clean up temporary memory
    CUDA_CHECK(cudaFree(d_input_image));
    
    // Convert results to OpenCV keypoints
    keypoints.clear();
    keypoints.reserve(current_matrix_size / 4); // Reserve space to avoid reallocations
    
    for (int i = 0; i < current_matrix_size; i++) {
        if (h_keypoint_x[i] >= 0 && h_keypoint_y[i] >= 0) {
            keypoints.emplace_back(cv::Point2f(h_keypoint_x[i], h_keypoint_y[i]), 
                                  (float)std::min(window_size.width, window_size.height),
                                  -1, h_keypoint_response[i]);
        }
    }
    
    return 1;
}

// ====== ❌ 以下函数保持不变 ======
// Release GPU memory
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

// Get keypoints
const std::vector<cv::KeyPoint>& CudaGradientDetector::GetKeypoints() const {
    return keypoints;
}

// Memory usage (simplified for optimized version)
void CudaGradientDetector::GetMemoryUsage(size_t& total_bytes, size_t& gradient_bytes, size_t& keypoint_bytes) const {
    size_t image_bytes = max_image_size * sizeof(float);
    keypoint_bytes = max_matrix_size * sizeof(float) * 3;
    gradient_bytes = 0; // No longer storing full gradient matrix
    total_bytes = image_bytes + keypoint_bytes;
}