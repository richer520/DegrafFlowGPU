#include "inference/RaftEngineTRT.h"

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#if DEGRAF_HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>
#endif

namespace
{
std::string envOrDefault(const char *name, const std::string &fallback)
{
    const char *value = std::getenv(name);
    if (value && std::string(value).size() > 0)
        return std::string(value);
    return fallback;
}

bool envEnabled(const char *name, bool default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    std::string v(value);
    for (char &c : v)
        c = static_cast<char>(std::tolower(c));
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

std::string toLower(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(c));
    return s;
}

float bilinearSample(const std::vector<float> &chw,
                     int channel,
                     int h, int w,
                     float y, float x)
{
    const int x0 = std::max(0, std::min(w - 1, static_cast<int>(std::floor(x))));
    const int x1 = std::max(0, std::min(w - 1, x0 + 1));
    const int y0 = std::max(0, std::min(h - 1, static_cast<int>(std::floor(y))));
    const int y1 = std::max(0, std::min(h - 1, y0 + 1));

    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const auto at = [&](int yy, int xx) -> float {
        const size_t idx = static_cast<size_t>(channel) * h * w + static_cast<size_t>(yy) * w + xx;
        return chw[idx];
    };

    const float v00 = at(y0, x0);
    const float v01 = at(y0, x1);
    const float v10 = at(y1, x0);
    const float v11 = at(y1, x1);
    const float v0 = v00 * (1.0f - wx) + v01 * wx;
    const float v1 = v10 * (1.0f - wx) + v11 * wx;
    return v0 * (1.0f - wy) + v1 * wy;
}

bool runLkFallback(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<std::vector<cv::Point2f>> &batch_points,
    std::vector<SparseFlowMatches> &batch_matches)
{
    batch_matches.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_points.size())
        return false;

    std::cout << "[PROFILE][RaftEngineTRT] backend=lk_fallback"
              << " frames=" << batch_i1.size() << std::endl;

    batch_matches.reserve(batch_i1.size());
    double total_lk_ms = 0.0;

    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        cv::Mat gray1, gray2;
        if (batch_i1[i].channels() == 3)
            cv::cvtColor(batch_i1[i], gray1, cv::COLOR_BGR2GRAY);
        else
            gray1 = batch_i1[i];

        if (batch_i2[i].channels() == 3)
            cv::cvtColor(batch_i2[i], gray2, cv::COLOR_BGR2GRAY);
        else
            gray2 = batch_i2[i];

        const auto &src = batch_points[i];
        std::vector<cv::Point2f> dst;
        std::vector<unsigned char> status;
        std::vector<float> err;

        auto lk_start = std::chrono::high_resolution_clock::now();
        if (!src.empty())
        {
            cv::calcOpticalFlowPyrLK(
                gray1,
                gray2,
                src,
                dst,
                status,
                err,
                cv::Size(11, 11),
                4);
        }
        auto lk_end = std::chrono::high_resolution_clock::now();
        const double lk_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(lk_end - lk_start).count();
        total_lk_ms += lk_ms;

        SparseFlowMatches matches;
        matches.src_points.reserve(src.size());
        matches.dst_points.reserve(src.size());

        const int max_flow_length = 100;
        for (size_t p = 0; p < src.size(); ++p)
        {
            if (p >= dst.size() || p >= status.size() || !status[p])
                continue;

            const cv::Point2f &s = src[p];
            const cv::Point2f &d = dst[p];
            const float dx = s.x - d.x;
            const float dy = s.y - d.y;
            if (std::sqrt(dx * dx + dy * dy) >= max_flow_length)
                continue;

            if (d.x < 0 || d.y < 0 || d.x >= gray2.cols || d.y >= gray2.rows)
                continue;
            if (s.x < 0 || s.y < 0 || s.x >= gray1.cols || s.y >= gray1.rows)
                continue;

            matches.src_points.push_back(s);
            matches.dst_points.push_back(d);
        }

        std::cout << "[PROFILE][RaftEngineTRT][Frame " << i << "] src_points=" << src.size()
                  << " matched_points=" << matches.dst_points.size()
                  << " lk_ms=" << lk_ms
                  << std::endl;

        batch_matches.push_back(std::move(matches));
    }

    std::cout << "[PROFILE][RaftEngineTRT] backend=lk_fallback"
              << " total_lk_ms=" << total_lk_ms << std::endl;
    return true;
}

#if DEGRAF_HAVE_TENSORRT
class TrtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > Severity::kWARNING)
            return;
        std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

class RaftTrtRunner
{
public:
    bool init(const std::string &engine_path)
    {
        if (initialized_ && engine_path_ == engine_path)
            return true;

        reset();
        engine_path_ = engine_path;

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.is_open())
            return false;
        file.seekg(0, std::ios::end);
        const std::streamoff size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (size <= 0)
            return false;

        std::vector<char> engine_data(static_cast<size_t>(size));
        file.read(engine_data.data(), size);
        if (!file.good())
            return false;

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
            return false;
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_)
            return false;
        context_.reset(engine_->createExecutionContext());
        if (!context_)
            return false;

        if (!resolveBindings())
            return false;

        initialized_ = true;
        return true;
    }

    bool inferDenseFlow(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &flow_hw2)
    {
        if (!initialized_)
            return false;
        if (img1.empty() || img2.empty() || img1.size() != img2.size())
            return false;

        const int h = img1.rows;
        const int w = img1.cols;

        nvinfer1::Dims4 input_dims(1, 3, h, w);
        if (!context_->setBindingDimensions(input0_idx_, input_dims) ||
            !context_->setBindingDimensions(input1_idx_, input_dims))
            return false;
        if (!context_->allInputDimensionsSpecified())
            return false;

        const nvinfer1::Dims out_dims = context_->getBindingDimensions(output_idx_);
        if (out_dims.nbDims != 4 || out_dims.d[1] < 2)
            return false;
        const int out_h = out_dims.d[2];
        const int out_w = out_dims.d[3];
        if (out_h <= 0 || out_w <= 0)
            return false;

        const size_t in_elems = static_cast<size_t>(3) * h * w;
        const size_t out_channels = static_cast<size_t>(out_dims.d[1]);
        const size_t out_elems = out_channels * out_h * out_w;

        std::vector<float> in0(in_elems);
        std::vector<float> in1(in_elems);
        std::vector<float> out(out_elems);
        toCHWFloat(img1, in0);
        toCHWFloat(img2, in1);

        void *bindings[3] = {nullptr, nullptr, nullptr};
        if (!allocDevice(input0_idx_, in_elems * sizeof(float), bindings[input0_idx_]) ||
            !allocDevice(input1_idx_, in_elems * sizeof(float), bindings[input1_idx_]) ||
            !allocDevice(output_idx_, out_elems * sizeof(float), bindings[output_idx_]))
            return false;

        if (cudaMemcpy(bindings[input0_idx_], in0.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(bindings[input1_idx_], in1.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return false;

        if (!context_->enqueueV2(bindings, 0, nullptr))
            return false;

        if (cudaMemcpy(out.data(), bindings[output_idx_], out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;

        flow_hw2 = cv::Mat(out_h, out_w, CV_32FC2, cv::Scalar(0, 0));
        for (int yy = 0; yy < out_h; ++yy)
        {
            cv::Vec2f *row = flow_hw2.ptr<cv::Vec2f>(yy);
            for (int xx = 0; xx < out_w; ++xx)
            {
                const size_t base = static_cast<size_t>(yy) * out_w + xx;
                row[xx][0] = out[base];
                row[xx][1] = out[static_cast<size_t>(out_h) * out_w + base];
            }
        }
        return true;
    }

private:
    struct TRTDestroy
    {
        template <typename T>
        void operator()(T *obj) const
        {
            if (obj)
                obj->destroy();
        }
    };

    void reset()
    {
        for (void *ptr : device_buffers_)
        {
            if (ptr)
                cudaFree(ptr);
        }
        device_buffers_.assign(3, nullptr);
        allocated_bytes_.assign(3, 0);
        context_.reset();
        engine_.reset();
        runtime_.reset();
        initialized_ = false;
        input0_idx_ = input1_idx_ = output_idx_ = -1;
    }

    bool resolveBindings()
    {
        const int nb = engine_->getNbBindings();
        if (nb < 3)
            return false;

        device_buffers_.assign(static_cast<size_t>(nb), nullptr);
        allocated_bytes_.assign(static_cast<size_t>(nb), 0);

        for (int i = 0; i < nb; ++i)
        {
            const std::string name = toLower(engine_->getBindingName(i));
            if (engine_->bindingIsInput(i))
            {
                if (name.find("image1") != std::string::npos || (input0_idx_ < 0 && name.find("input") != std::string::npos))
                    input0_idx_ = i;
                else if (name.find("image2") != std::string::npos || input1_idx_ < 0)
                    input1_idx_ = i;
            }
            else
            {
                if (name.find("flow_up") != std::string::npos || output_idx_ < 0)
                    output_idx_ = i;
            }
        }

        return input0_idx_ >= 0 && input1_idx_ >= 0 && output_idx_ >= 0;
    }

    bool allocDevice(int binding_idx, size_t bytes, void *&out_ptr)
    {
        if (binding_idx < 0 || static_cast<size_t>(binding_idx) >= device_buffers_.size())
            return false;
        if (device_buffers_[binding_idx] && allocated_bytes_[binding_idx] >= bytes)
        {
            out_ptr = device_buffers_[binding_idx];
            return true;
        }

        if (device_buffers_[binding_idx])
            cudaFree(device_buffers_[binding_idx]);

        void *ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess)
            return false;

        device_buffers_[binding_idx] = ptr;
        allocated_bytes_[binding_idx] = bytes;
        out_ptr = ptr;
        return true;
    }

    static void toCHWFloat(const cv::Mat &img, std::vector<float> &out)
    {
        cv::Mat rgb;
        if (img.channels() == 3)
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        else
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);

        cv::Mat f32;
        rgb.convertTo(f32, CV_32FC3);

        const int h = f32.rows;
        const int w = f32.cols;
        out.resize(static_cast<size_t>(3) * h * w);
        for (int y = 0; y < h; ++y)
        {
            const cv::Vec3f *row = f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < w; ++x)
            {
                const size_t hw = static_cast<size_t>(y) * w + x;
                out[hw] = row[x][0];
                out[static_cast<size_t>(h) * w + hw] = row[x][1];
                out[2ull * h * w + hw] = row[x][2];
            }
        }
    }

private:
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy> context_;
    std::vector<void *> device_buffers_;
    std::vector<size_t> allocated_bytes_;

    std::string engine_path_;
    bool initialized_ = false;
    int input0_idx_ = -1;
    int input1_idx_ = -1;
    int output_idx_ = -1;
};
#endif
} // namespace

bool RaftEngineTRT::estimateMatchesBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<std::vector<cv::Point2f>> &batch_points,
    std::vector<SparseFlowMatches> &batch_matches)
{
    const std::string backend = envOrDefault("DEGRAF_RAFT_BACKEND", "trt");
    const bool allow_lk_fallback = envEnabled("DEGRAF_ALLOW_LK_FALLBACK", false);

    if (backend == "lk" || backend == "lk_fallback")
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);

    if (backend != "trt")
    {
        std::cerr << "[ERROR][RaftEngineTRT] Unsupported backend: " << backend
                  << " (expected trt or lk)" << std::endl;
        return false;
    }

    const std::string engine_path = envOrDefault("DEGRAF_RAFT_ENGINE_PATH", "");
    std::cout << "[PROFILE][RaftEngineTRT] backend=trt"
              << " frames=" << batch_i1.size() << std::endl;

    if (engine_path.empty() || !cv::utils::fs::exists(engine_path))
    {
        std::cerr << "[ERROR][RaftEngineTRT] Missing or invalid DEGRAF_RAFT_ENGINE_PATH: "
                  << (engine_path.empty() ? "<empty>" : engine_path) << std::endl;
        if (!allow_lk_fallback)
            return false;
        std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
    }

#if DEGRAF_HAVE_TENSORRT
    static RaftTrtRunner runner;
    if (!runner.init(engine_path))
    {
        std::cerr << "[ERROR][RaftEngineTRT] Failed to initialize TensorRT engine: " << engine_path << std::endl;
        if (!allow_lk_fallback)
            return false;
        std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
        return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
    }

    batch_matches.clear();
    batch_matches.reserve(batch_i1.size());
    double total_trt_ms = 0.0;
    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        const auto &src = batch_points[i];
        SparseFlowMatches matches;
        matches.src_points.reserve(src.size());
        matches.dst_points.reserve(src.size());

        cv::Mat dense_flow;
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!runner.inferDenseFlow(batch_i1[i], batch_i2[i], dense_flow))
        {
            std::cerr << "[ERROR][RaftEngineTRT] TensorRT inference failed on frame " << i << std::endl;
            if (!allow_lk_fallback)
                return false;
            std::cerr << "[WARN][RaftEngineTRT] TensorRT failed; switching whole batch to LK fallback." << std::endl;
            return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double trt_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
        total_trt_ms += trt_ms;

        std::vector<float> flow_chw(static_cast<size_t>(2) * dense_flow.rows * dense_flow.cols, 0.0f);
        for (int yy = 0; yy < dense_flow.rows; ++yy)
        {
            const cv::Vec2f *row = dense_flow.ptr<cv::Vec2f>(yy);
            for (int xx = 0; xx < dense_flow.cols; ++xx)
            {
                const size_t hw = static_cast<size_t>(yy) * dense_flow.cols + xx;
                flow_chw[hw] = row[xx][0];
                flow_chw[static_cast<size_t>(dense_flow.rows) * dense_flow.cols + hw] = row[xx][1];
            }
        }

        const float sx = static_cast<float>(dense_flow.cols) / static_cast<float>(batch_i1[i].cols);
        const float sy = static_cast<float>(dense_flow.rows) / static_cast<float>(batch_i1[i].rows);
        const int max_flow_length = 100;
        for (const auto &p : src)
        {
            const float fx = p.x * sx;
            const float fy = p.y * sy;
            if (fx < 0 || fy < 0 || fx >= dense_flow.cols || fy >= dense_flow.rows)
                continue;

            const float du = bilinearSample(flow_chw, 0, dense_flow.rows, dense_flow.cols, fy, fx);
            const float dv = bilinearSample(flow_chw, 1, dense_flow.rows, dense_flow.cols, fy, fx);

            const cv::Point2f d(p.x + du, p.y + dv);
            const float dx = p.x - d.x;
            const float dy = p.y - d.y;
            if (std::sqrt(dx * dx + dy * dy) >= max_flow_length)
                continue;
            if (d.x < 0 || d.y < 0 || d.x >= batch_i2[i].cols || d.y >= batch_i2[i].rows)
                continue;
            matches.src_points.push_back(p);
            matches.dst_points.push_back(d);
        }

        std::cout << "[PROFILE][RaftEngineTRT][Frame " << i << "] src_points=" << src.size()
                  << " matched_points=" << matches.dst_points.size()
                  << " trt_ms=" << trt_ms << std::endl;
        batch_matches.push_back(std::move(matches));
    }
    std::cout << "[PROFILE][RaftEngineTRT] backend=trt total_trt_ms=" << total_trt_ms << std::endl;
    return true;
#else
    std::cerr << "[ERROR][RaftEngineTRT] This binary was built without TensorRT support (DEGRAF_HAVE_TENSORRT=0)."
              << " Rebuild with TensorRT enabled." << std::endl;
#endif
    if (!allow_lk_fallback)
        return false;
    std::cerr << "[WARN][RaftEngineTRT] Falling back to LK because DEGRAF_ALLOW_LK_FALLBACK=1" << std::endl;
    return runLkFallback(batch_i1, batch_i2, batch_points, batch_matches);
}

