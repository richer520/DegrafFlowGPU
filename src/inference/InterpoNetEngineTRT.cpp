#include "inference/InterpoNetEngineTRT.h"

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>

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

bool runVariationalRefine(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dense_flow, size_t idx)
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

    const std::string cmd =
        "\"" + variational_bin + "\" " +
        "\"" + img1_path + "\" " +
        "\"" + img2_path + "\" " +
        "\"" + flow_in_path + "\" " +
        "\"" + flow_out_path + "\" -" + dataset;

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

    batch_flows.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_matches.size())
        return false;

    batch_flows.reserve(batch_i1.size());

    for (size_t i = 0; i < batch_i1.size(); ++i)
    {
        cv::Mat dense_flow(batch_i1[i].size(), CV_32FC2, cv::Scalar(0, 0));

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

        if (enable_variational && !dense_flow.empty())
            runVariationalRefine(batch_i1[i], batch_i2[i], dense_flow, i);

        batch_flows.push_back(dense_flow);
    }

    return true;
}

