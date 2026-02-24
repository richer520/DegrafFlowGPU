#include "inference/RaftEngineTRT.h"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

bool RaftEngineTRT::estimateMatchesBatch(
    const std::vector<cv::Mat> &batch_i1,
    const std::vector<cv::Mat> &batch_i2,
    const std::vector<std::vector<cv::Point2f>> &batch_points,
    std::vector<SparseFlowMatches> &batch_matches)
{
    batch_matches.clear();
    if (batch_i1.size() != batch_i2.size() || batch_i1.size() != batch_points.size())
        return false;

    batch_matches.reserve(batch_i1.size());

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

        batch_matches.push_back(std::move(matches));
    }

    return true;
}

