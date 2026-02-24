#include "inference/InterpoNetEngineTRT.h"

#include <opencv2/ximgproc/sparse_match_interpolator.hpp>

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

        batch_flows.push_back(dense_flow);
    }

    return true;
}

