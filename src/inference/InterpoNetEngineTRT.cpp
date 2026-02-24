#include "inference/InterpoNetEngineTRT.h"

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
#include "image.h"
#include "variational.h"
#endif

namespace
{
std::string getProjectRoot()
{
    const char *env_path = std::getenv("DEGRAF_PROJECT_ROOT");
    if (env_path && std::string(env_path).size() > 0)
        return std::string(env_path);
    if (cv::utils::fs::exists("/root/autodl-tmp/projects/DegrafFlowGPU"))
        return "/root/autodl-tmp/projects/DegrafFlowGPU";
    return ".";
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

std::string envOrDefault(const char *name, const std::string &fallback)
{
    const char *value = std::getenv(name);
    if (value && std::string(value).size() > 0)
        return std::string(value);
    return fallback;
}

bool envInt(const char *name, int &out_value)
{
    const char *value = std::getenv(name);
    if (!value || std::string(value).empty())
        return false;
    try
    {
        out_value = std::stoi(value);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool envFloat(const char *name, float &out_value)
{
    const char *value = std::getenv(name);
    if (!value || std::string(value).empty())
        return false;
    try
    {
        out_value = std::stof(value);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

double computeSparseResidualPx(const cv::Mat &dense_flow, const SparseFlowMatches &matches)
{
    if (dense_flow.empty() || dense_flow.type() != CV_32FC2)
        return -1.0;
    if (matches.src_points.empty() || matches.src_points.size() != matches.dst_points.size())
        return -1.0;

    double sum_epe = 0.0;
    int used = 0;
    for (size_t i = 0; i < matches.src_points.size(); ++i)
    {
        const cv::Point2f &s = matches.src_points[i];
        const cv::Point2f &d = matches.dst_points[i];
        const int x = static_cast<int>(std::round(s.x));
        const int y = static_cast<int>(std::round(s.y));
        if (x < 0 || y < 0 || x >= dense_flow.cols || y >= dense_flow.rows)
            continue;

        const cv::Vec2f &f = dense_flow.at<cv::Vec2f>(y, x);
        const float px = s.x + f[0];
        const float py = s.y + f[1];
        const float ex = px - d.x;
        const float ey = py - d.y;
        sum_epe += std::sqrt(ex * ex + ey * ey);
        ++used;
    }
    if (used == 0)
        return -1.0;
    return sum_epe / static_cast<double>(used);
}

bool writeFlowFile(const std::string &path, const cv::Mat &flow)
{
    if (flow.empty() || flow.type() != CV_32FC2)
        return false;

    std::ofstream file(path.c_str(), std::ios::binary);
    if (!file.good())
        return false;

    const float magic = 202021.25f;
    const int width = flow.cols;
    const int height = flow.rows;
    file.write(reinterpret_cast<const char *>(&magic), sizeof(float));
    file.write(reinterpret_cast<const char *>(&width), sizeof(int));
    file.write(reinterpret_cast<const char *>(&height), sizeof(int));
    file.write(reinterpret_cast<const char *>(flow.data), width * height * 2 * sizeof(float));
    return file.good();
}

cv::Mat readFlowFile(const std::string &path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file.good())
        return cv::Mat();

    float magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(float));
    if (magic != 202021.25f)
        return cv::Mat();

    int width = 0, height = 0;
    file.read(reinterpret_cast<char *>(&width), sizeof(int));
    file.read(reinterpret_cast<char *>(&height), sizeof(int));
    if (width <= 0 || height <= 0)
        return cv::Mat();

    cv::Mat flow(height, width, CV_32FC2);
    file.read(reinterpret_cast<char *>(flow.data), width * height * 2 * sizeof(float));
    if (!file.good())
        return cv::Mat();
    return flow;
}

bool runVariationalRefineExternal(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow, size_t idx)
{
    const std::string project_root = getProjectRoot();
    const std::string variational_bin =
        envOrDefault("DEGRAF_VARIATIONAL_BIN",
                     project_root + "/external/InterpoNet/SrcVariational/variational_main");
    if (!cv::utils::fs::exists(variational_bin))
        return false;

    const std::string tmp_dir =
        envOrDefault("DEGRAF_VARIATIONAL_TMP_DIR", project_root + "/data/outputs/variational_tmp");
    cv::utils::fs::createDirectories(tmp_dir);

    const std::string dataset = envOrDefault("DEGRAF_VARIATIONAL_DATASET", "kitti");
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string token = std::to_string(static_cast<long long>(stamp)) + "_" + std::to_string(idx);

    const std::string img1_path = tmp_dir + "/" + token + "_a.png";
    const std::string img2_path = tmp_dir + "/" + token + "_b.png";
    const std::string flow_in_path = tmp_dir + "/" + token + "_in.flo";
    const std::string flow_out_path = tmp_dir + "/" + token + "_out.flo";

    if (!cv::imwrite(img1_path, img1) || !cv::imwrite(img2_path, img2))
        return false;
    if (!writeFlowFile(flow_in_path, dense_flow))
        return false;

    std::ostringstream args;
    int niter_outer = 0;
    float alpha = 0.0f, gamma = 0.0f, delta = 0.0f, sigma = 0.0f;
    if (envInt("DEGRAF_VARIATIONAL_NITER_OUTER", niter_outer))
        args << " -iter " << niter_outer;
    if (envFloat("DEGRAF_VARIATIONAL_ALPHA", alpha))
        args << " -alpha " << alpha;
    if (envFloat("DEGRAF_VARIATIONAL_GAMMA", gamma))
        args << " -gamma " << gamma;
    if (envFloat("DEGRAF_VARIATIONAL_DELTA", delta))
        args << " -delta " << delta;
    if (envFloat("DEGRAF_VARIATIONAL_SIGMA", sigma))
        args << " -sigma " << sigma;

    const std::string cmd =
        "\"" + variational_bin + "\" " +
        "\"" + img1_path + "\" " +
        "\"" + img2_path + "\" " +
        "\"" + flow_in_path + "\" " +
        "\"" + flow_out_path + "\" -" + dataset + args.str();

    const int exit_code = std::system(cmd.c_str());
    if (exit_code != 0)
        return false;

    cv::Mat refined = readFlowFile(flow_out_path);
    if (refined.empty() || refined.size() != dense_flow.size() || refined.type() != dense_flow.type())
        return false;

    dense_flow = refined;

    // Keep temp files only when explicitly requested for debugging.
    if (!envEnabled("DEGRAF_KEEP_VARIATIONAL_TMP", false))
    {
        std::remove(img1_path.c_str());
        std::remove(img2_path.c_str());
        std::remove(flow_in_path.c_str());
        std::remove(flow_out_path.c_str());
    }
    return true;
}

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
color_image_t *toVariationalColorImage(const cv::Mat &input)
{
    cv::Mat bgr;
    if (input.channels() == 1)
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    else if (input.channels() == 4)
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
    else
        bgr = input;

    cv::Mat bgr_u8;
    if (bgr.type() != CV_8UC3)
        bgr.convertTo(bgr_u8, CV_8UC3);
    else
        bgr_u8 = bgr;

    color_image_t *out = color_image_new(bgr_u8.cols, bgr_u8.rows);
    for (int y = 0; y < bgr_u8.rows; ++y)
    {
        const cv::Vec3b *row = bgr_u8.ptr<cv::Vec3b>(y);
        const int base = y * out->stride;
        for (int x = 0; x < bgr_u8.cols; ++x)
        {
            // variational code expects RGB layout in c1/c2/c3.
            out->c1[base + x] = static_cast<float>(row[x][2]);
            out->c2[base + x] = static_cast<float>(row[x][1]);
            out->c3[base + x] = static_cast<float>(row[x][0]);
        }
    }
    return out;
}

void flowMatToVariationalImages(const cv::Mat &flow, image_t *wx, image_t *wy)
{
    for (int y = 0; y < flow.rows; ++y)
    {
        const cv::Vec2f *row = flow.ptr<cv::Vec2f>(y);
        const int base = y * wx->stride;
        for (int x = 0; x < flow.cols; ++x)
        {
            wx->data[base + x] = row[x][0];
            wy->data[base + x] = row[x][1];
        }
    }
}

cv::Mat variationalImagesToFlowMat(const image_t *wx, const image_t *wy)
{
    cv::Mat flow(wx->height, wx->width, CV_32FC2, cv::Scalar(0, 0));
    for (int y = 0; y < wx->height; ++y)
    {
        cv::Vec2f *row = flow.ptr<cv::Vec2f>(y);
        const int base = y * wx->stride;
        for (int x = 0; x < wx->width; ++x)
        {
            row[x][0] = wx->data[base + x];
            row[x][1] = wy->data[base + x];
        }
    }
    return flow;
}

void applyPresetFromDataset(variational_params_t &params, const std::string &dataset)
{
    if (dataset == "sintel")
    {
        params.niter_outer = 5;
        params.alpha = 1.0f;
        params.gamma = 0.72f;
        params.delta = 0.0f;
        params.sigma = 1.1f;
    }
    else if (dataset == "middlebury")
    {
        params.niter_outer = 25;
        params.alpha = 1.0f;
        params.gamma = 0.72f;
        params.delta = 0.0f;
        params.sigma = 1.1f;
    }
    else
    {
        // KITTI preset from original variational_main.
        params.niter_outer = 2;
        params.alpha = 1.0f;
        params.gamma = 0.77f;
        params.delta = 0.0f;
        params.sigma = 1.7f;
    }
}

void applyVariationalOverridesFromEnv(variational_params_t &params)
{
    int niter_outer = 0, niter_inner = 0, niter_solver = 0;
    float alpha = 0.0f, gamma = 0.0f, delta = 0.0f, sigma = 0.0f, sor_omega = 0.0f;

    if (envInt("DEGRAF_VARIATIONAL_NITER_OUTER", niter_outer) && niter_outer > 0)
        params.niter_outer = niter_outer;
    if (envInt("DEGRAF_VARIATIONAL_NITER_INNER", niter_inner) && niter_inner > 0)
        params.niter_inner = niter_inner;
    if (envInt("DEGRAF_VARIATIONAL_NITER_SOLVER", niter_solver) && niter_solver > 0)
        params.niter_solver = niter_solver;
    if (envFloat("DEGRAF_VARIATIONAL_ALPHA", alpha) && alpha >= 0.0f)
        params.alpha = alpha;
    if (envFloat("DEGRAF_VARIATIONAL_GAMMA", gamma) && gamma >= 0.0f)
        params.gamma = gamma;
    if (envFloat("DEGRAF_VARIATIONAL_DELTA", delta) && delta >= 0.0f)
        params.delta = delta;
    if (envFloat("DEGRAF_VARIATIONAL_SIGMA", sigma) && sigma > 0.0f)
        params.sigma = sigma;
    if (envFloat("DEGRAF_VARIATIONAL_SOR_OMEGA", sor_omega) && sor_omega > 0.0f)
        params.sor_omega = sor_omega;
}

bool runVariationalRefineInProcess(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow)
{
    if (dense_flow.empty() || dense_flow.type() != CV_32FC2)
        return false;

    color_image_t *im1 = toVariationalColorImage(img1);
    color_image_t *im2 = toVariationalColorImage(img2);
    image_t *wx = image_new(dense_flow.cols, dense_flow.rows);
    image_t *wy = image_new(dense_flow.cols, dense_flow.rows);

    flowMatToVariationalImages(dense_flow, wx, wy);

    variational_params_t params;
    variational_params_default(&params);
    applyPresetFromDataset(params, envOrDefault("DEGRAF_VARIATIONAL_DATASET", "kitti"));
    applyVariationalOverridesFromEnv(params);
    variational(wx, wy, im1, im2, &params);

    dense_flow = variationalImagesToFlowMat(wx, wy);

    color_image_delete(im1);
    color_image_delete(im2);
    image_delete(wx);
    image_delete(wy);
    return true;
}
#endif

bool runVariationalRefine(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow, size_t idx)
{
    const std::string mode = envOrDefault("DEGRAF_VARIATIONAL_MODE", "external");
    if (mode == "external")
        return runVariationalRefineExternal(img1, img2, dense_flow, idx);

#if DEGRAF_HAVE_INPROCESS_VARIATIONAL
    if (runVariationalRefineInProcess(img1, img2, dense_flow))
        return true;
#endif
    return runVariationalRefineExternal(img1, img2, dense_flow, idx);
}
} // namespace

InterpoNetEngineTRT::InterpoNetEngineTRT(int k, float sigma, bool use_post_proc,
                                         float fgs_lambda, float fgs_sigma)
    : k_(k),
      sigma_(sigma),
      use_post_proc_(use_post_proc),
      fgs_lambda_(fgs_lambda),
      fgs_sigma_(fgs_sigma)
{
}

bool InterpoNetEngineTRT::densifyBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<SparseFlowMatches> &batch_matches,
    std::vector<cv::Mat> &batch_flows)
{
    const bool enable_variational = envEnabled("DEGRAF_ENABLE_VARIATIONAL", true);
    const bool profile_stages = envEnabled("DEGRAF_PROFILE_STAGES", true);
    const bool residual_gate = envEnabled("DEGRAF_VARIATIONAL_RESIDUAL_GATE", false);
    float residual_thresh = 2.0f;
    envFloat("DEGRAF_VARIATIONAL_RESIDUAL_THRESH", residual_thresh);

    batch_flows.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_matches.size())
        return false;

    batch_flows.reserve(batch_i1.size());
    double total_interpolate_ms = 0.0;
    double total_variational_ms = 0.0;
    int variational_applied = 0;

    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        cv::Mat dense_flow(batch_i1[i].size(), CV_32FC2, cv::Scalar(0, 0));

        auto interp_start = std::chrono::high_resolution_clock::now();
        if (!batch_matches[i].src_points.empty() &&
            batch_matches[i].src_points.size() == batch_matches[i].dst_points.size())
        {
            cv::Ptr<cv::ximgproc::EdgeAwareInterpolator> interpolator =
                cv::ximgproc::createEdgeAwareInterpolator();
            interpolator->setK(k_);
            interpolator->setSigma(sigma_);
            interpolator->setUsePostProcessing(use_post_proc_);
            interpolator->setFGSLambda(fgs_lambda_);
            interpolator->setFGSSigma(fgs_sigma_);
            interpolator->interpolate(
                batch_i1[i],
                batch_matches[i].src_points,
                batch_i2[i],
                batch_matches[i].dst_points,
                dense_flow);
        }
        auto interp_end = std::chrono::high_resolution_clock::now();
        total_interpolate_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(interp_end - interp_start).count();

        bool should_run_variational = enable_variational && !dense_flow.empty();
        double sparse_residual_px = -1.0;
        if (should_run_variational && residual_gate)
        {
            sparse_residual_px = computeSparseResidualPx(dense_flow, batch_matches[i]);
            if (sparse_residual_px >= 0.0 && sparse_residual_px < residual_thresh)
                should_run_variational = false;
        }

        if (should_run_variational)
        {
            auto vari_start = std::chrono::high_resolution_clock::now();
            runVariationalRefine(batch_i1[i], batch_i2[i], dense_flow, i);
            auto vari_end = std::chrono::high_resolution_clock::now();
            total_variational_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(vari_end - vari_start).count();
            ++variational_applied;
        }

        if (profile_stages && residual_gate)
        {
            std::cout << "[PROFILE][InterpoNetEngineTRT][Frame " << i << "] residual_px=" << sparse_residual_px
                      << " thresh=" << residual_thresh
                      << " variational=" << (should_run_variational ? "on" : "skip")
                      << std::endl;
        }

        batch_flows.push_back(dense_flow);
    }

    if (profile_stages)
    {
        std::cout << "[PROFILE][InterpoNetEngineTRT] frames=" << batch_i1.size()
                  << " interpolate_ms=" << total_interpolate_ms
                  << " variational_ms=" << total_variational_ms
                  << " variational_frames=" << variational_applied
                  << " total_ms=" << (total_interpolate_ms + total_variational_ms)
                  << std::endl;
    }

    return true;
}

