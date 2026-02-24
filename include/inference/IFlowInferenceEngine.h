#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct SparseFlowMatches
{
    std::vector<cv::Point2f> src_points;
    std::vector<cv::Point2f> dst_points;
};

class IRaftEngine
{
public:
    virtual ~IRaftEngine() = default;

    virtual bool estimateMatchesBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<std::vector<cv::Point2f>> &batch_points,
        std::vector<SparseFlowMatches> &batch_matches) = 0;
};

class IInterpoNetEngine
{
public:
    virtual ~IInterpoNetEngine() = default;

    virtual bool densifyBatch(
        const std::vector<cv::Mat> &batch_i1,
        const std::vector<cv::Mat> &batch_i2,
        const std::vector<SparseFlowMatches> &batch_matches,
        std::vector<cv::Mat> &batch_flows) = 0;
};

