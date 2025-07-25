/*!
\file EvaluateSceneFlow.cpp
\brief Scene Flow evaluation implementation - 严格参考EvaluateOptFlow设计风格
\author Gang Wang, Durham University
*/

#include "EvaluateSceneFlow.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/image_viewer.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/common/time.h>

using namespace cv;
using namespace std;

EvaluateSceneFlow::EvaluateSceneFlow() {}
// 集中定义所有辅助函数在文件顶部

/**
 * 功能：从 KITTI calib_cam_to_cam/*.txt 文件中解析 fx, fy, cx, cy, baseline
 * 输入参数：
 * calib_file: 文件路径
 * fx, fy, cx, cy, baseline: 引用变量，写入解析值
 * 输出：true 表示成功解析，false 表示失败或格式不对
 */
static bool loadCameraIntrinsics(const std::string &calib_file, float &fx, float &fy, float &cx, float &cy, float &baseline)
{
    std::ifstream file(calib_file);
    if (!file.is_open())
    {
        return false;
    }

    std::string line;
    std::vector<float> P2_values;
    std::vector<float> P3_values;

    while (std::getline(file, line))
    {
        if (line.find("P_rect_02:") != std::string::npos)
        {
            std::istringstream iss(line.substr(11)); // 跳过标签
            float val;
            while (iss >> val)
            {
                P2_values.push_back(val);
            }
        }
        else if (line.find("P_rect_03:") != std::string::npos)
        {
            std::istringstream iss(line.substr(11));
            float val;
            while (iss >> val)
            {
                P3_values.push_back(val);
            }
        }
    }
    file.close();

    if (P2_values.size() < 12 || P3_values.size() < 12)
    {
        return false;
    }

    // 从 P2 解析内参
    fx = P2_values[0]; // P2[0][0]
    fy = P2_values[5]; // P2[1][1]
    cx = P2_values[2]; // P2[0][2]
    cy = P2_values[6]; // P2[1][2]

    // 从 P2[0][3] 和 P3[0][3] 计算 baseline
    float Tx2 = P2_values[3]; // P2[0][3]
    float Tx3 = P3_values[3]; // P3[0][3]
    baseline = -(Tx3 - Tx2) / fx;

    return true;
}

/**
 * @brief 生成最终评估掩码：仅在GT有效且满足指定区域的像素设为255，其余为0
 * @param gt_scene_flow  Ground truth三维场景流（CV_32FC3，单位米，NaN为无效）
 * @param image          原始输入图像（BGR或GRAY，用于untextured区域mask）
 * @param region         区域类型："all"（全图）、"discontinuities"（边缘）、"untextured"（低纹理区）
 * @return               CV_8UC1 掩码（255为有效像素，其余为无效）
 *
 * 实现要点说明
1.功能：
    保证只有“GT有效”+“满足区域条件”的像素被评估，所有统计、可视化都用此mask。
2.输入参数：
    gt_scene_flow：GT三维场景流，CV_32FC3，无效像素为NaN。
    image：原始图像，灰度或BGR，主要用于"untextured"。
    region：字符串，指定区域类型（"all"|"discontinuities"|"untextured"）。
3.输出结果：
    eval_mask：单通道8位二值图像，255表示该像素用于评估，0表示忽略。
4.细节与健壮性：
    "discontinuities"基于GT梯度自动适应场景流边界。
    "untextured"自动找出低纹理区域，兼容BGR/灰度输入。
    合并GT有效性与区域性，无需后续再and操作。
5.可扩展性：
    你以后可以轻松扩展其它mask类型，比如“foreground”、“occlusion”等。
 */
static Mat generateEvaluationMask(const cv::Mat &gt_scene_flow, const cv::Mat &image, const std::string &region)
{
    // 1. 生成valid_mask：GT中三通道都为有效数才算有效
    std::vector<cv::Mat> channels(3);
    cv::split(gt_scene_flow, channels);
    cv::Mat mask_x = channels[0] == channels[0];
    cv::Mat mask_y = channels[1] == channels[1];
    cv::Mat mask_z = channels[2] == channels[2];
    cv::Mat valid_mask = mask_x & mask_y & mask_z;
    valid_mask.convertTo(valid_mask, CV_8U, 255);

    // 2. 生成region_mask，默认全1
    cv::Mat region_mask = cv::Mat::ones(gt_scene_flow.size(), CV_8U) * 255;
    if (region == "discontinuities")
    {
        // 用GT场景流的三通道总和梯度表示边界区域
        cv::Mat merged = cv::abs(channels[0]) + cv::abs(channels[1]) + cv::abs(channels[2]);
        cv::Mat grad_x, grad_y, gradient;
        cv::Sobel(merged, grad_x, CV_16S, 1, 0);
        cv::Sobel(merged, grad_y, CV_16S, 0, 1);
        grad_x = cv::abs(grad_x);
        grad_y = cv::abs(grad_y);
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient);
        double mean_val = cv::mean(gradient)[0];
        region_mask = gradient > mean_val;
        cv::dilate(region_mask, region_mask, cv::Mat::ones(9, 9, CV_8U));
    }
    else if (region == "untextured")
    {
        // 用输入图像的低梯度区生成低纹理mask
        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image;
        cv::Mat grad_x, grad_y, gradient;
        cv::Sobel(gray, grad_x, CV_16S, 1, 0, 7);
        cv::Sobel(gray, grad_y, CV_16S, 0, 1, 7);
        grad_x = cv::abs(grad_x);
        grad_y = cv::abs(grad_y);
        cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient);
        cv::GaussianBlur(gradient, gradient, cv::Size(5, 5), 1, 1);
        region_mask = gradient < 30; // 低纹理阈值可自定义
        cv::dilate(region_mask, region_mask, cv::Mat::ones(3, 3, CV_8U));
    }
    // 3. 合并两种掩码：仅在GT有效且属于指定区域的像素才为255
    cv::Mat eval_mask;
    cv::bitwise_and(valid_mask, region_mask, eval_mask);
    return eval_mask;
}

/**
 * @brief 可视化3D场景流为伪彩色图像（方向+幅值），便于结果展示与主观对比。
 *
 * 实现要点说明：
 * - 输入为3D场景流（CV_32FC3，每像素x/y/z为单位米的运动向量），无效像素NaN可用mask或设黑色。
 * - 采用“色相=运动方向，饱和度/亮度=运动幅值”的HSV方案，将三维运动编码为颜色。
 * - 可选：Z轴（深度方向）可单独影响色调或亮度（这里直接取x/y分量方向）。
 * - 输出为BGR彩色图，可直接保存或imshow。
 * - 幅值上限（max_magnitude）可自定义，防止极大运动“溢出”。
 * - 无效像素（mask=0或NaN）直接设为黑色。
 *
 * @param scene_flow   CV_32FC3 预测或GT场景流
 * @param eval_mask    CV_8UC1 掩码（255=有效，0=无效），无掩码时可传空
 * @param max_magnitude 可视化最大运动幅值（如2.0m），超过部分设为最大
 * @return             CV_8UC3 彩色BGR显示图像
 */
static Mat sceneFlowToDisplay(const cv::Mat &scene_flow, const cv::Mat &eval_mask = cv::Mat(), float max_magnitude = 2.0f)
{
    CV_Assert(scene_flow.type() == CV_32FC3);

    int rows = scene_flow.rows, cols = scene_flow.cols;
    cv::Mat display(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0)); // 默认全黑

    for (int v = 0; v < rows; ++v)
    {
        for (int u = 0; u < cols; ++u)
        {
            // 掩码外设为黑色
            if (!eval_mask.empty() && eval_mask.at<uchar>(v, u) == 0)
                continue;

            cv::Vec3f flow = scene_flow.at<cv::Vec3f>(v, u);
            // 若有NaN或0向量，设为黑色
            if (!cv::checkRange(flow) || (flow[0] == 0 && flow[1] == 0 && flow[2] == 0))
                continue;

            // 计算2D平面运动方向和幅值（也可3D模长/夹角等扩展）
            float fx = flow[0], fy = flow[1], fz = flow[2];
            float magnitude = std::sqrt(fx * fx + fy * fy + fz * fz);

            // 用x/y分量计算方向（对比光流可视化风格）
            float angle = std::atan2(fy, fx);           // -pi~pi
            float hue = 90.0f - angle * 180.0f / CV_PI; // 色相角度，0=红，120=绿，240=蓝

            // 归一化幅值，最大设max_magnitude
            float norm_mag = std::min(magnitude / max_magnitude, 1.0f);

            // 组装HSV，S=255，V随幅值
            cv::Vec3b hsv;
            hsv[0] = static_cast<uchar>((hue < 0 ? hue + 360.0f : hue) / 2); // OpenCV: 0~180
            hsv[1] = 255;
            hsv[2] = static_cast<uchar>(norm_mag * 255.0f);

            cv::Vec3b bgr;
            cv::Mat tmp(1, 1, CV_8UC3, hsv);
            cv::cvtColor(tmp, tmp, cv::COLOR_HSV2BGR);
            bgr = tmp.at<cv::Vec3b>(0, 0);
            display.at<cv::Vec3b>(v, u) = bgr;
        }
    }
    return display;
}

/**
 * @brief 按照KITTI官方标准读取视差GT文件
 * @param path 视差文件路径（如disp_noc_0/000000_10.png）
 * @return CV_32FC1视差图，无效像素为NaN
 */
static cv::Mat readKITTIDisparity(const std::string &path)
{
    // 1. 读取原始uint16图像
    cv::Mat disp_raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (disp_raw.empty())
    {
        printf("❌ Cannot open disparity file: %s\n", path.c_str());
        return cv::Mat();
    }

    // 2. 检查类型：KITTI视差应该是CV_16UC1
    if (disp_raw.type() != CV_16UC1)
    {
        printf("❌ Invalid disparity image type (expect CV_16UC1, got %d)\n", disp_raw.type());
        return cv::Mat();
    }

    // 3. 按照KITTI标准转换
    cv::Mat disp_f32(disp_raw.size(), CV_32F);
    for (int y = 0; y < disp_raw.rows; ++y)
    {
        for (int x = 0; x < disp_raw.cols; ++x)
        {
            uint16_t raw_val = disp_raw.at<uint16_t>(y, x);
            if (raw_val == 0)
            {
                // ⭐️ KITTI标准：0值表示无效像素
                disp_f32.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                // ⭐️ KITTI标准：除以256得到真实视差值
                disp_f32.at<float>(y, x) = static_cast<float>(raw_val) / 256.0f;
            }
        }
    }
    return disp_f32;
}

/**
 * @brief 按照KITTI官方标准读取光流GT文件
 * @param path 光流文件路径（如flow_noc/000000_10.png或flow_occ/000000_10.png）
 * @return CV_32FC2光流图，无效像素为NaN
 */
/**
 * @brief 使用您原来的KITTI光流GT读取函数（经过验证的版本）
 */
static Mat readKITTIFlowGT(const std::string &ground_truth_path)
{
    String path = ground_truth_path;
    // NB opencv has order BGR => valid , v , u
    Mat image = imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    Mat gt = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
    int width = image.cols;
    int height = image.rows;
    int valid_pixels = 0; // 添加统计

    for (int32_t v = 0; v < height; v++)
    {
        for (int32_t u = 0; u < width; u++)
        {
            Vec3s val = image.at<Vec3s>(v, u);
            if (val[0] > 0) // 您原来的有效性判断
            {
                Vec2f flow;
                if (val[2] > 0)
                {
                    flow[0] = ((float)val[2] - 32768.0f) / 64.0f;
                }
                else
                {
                    flow[0] = ((float)val[2] + 32768.0f) / 64.0f;
                }

                if (val[1] > 0)
                {
                    flow[1] = ((float)val[1] - 32768.0f) / 64.0f;
                }
                else
                {
                    flow[1] = ((float)val[1] + 32768.0f) / 64.0f;
                }
                gt.at<Vec2f>(v, u) = flow;
                valid_pixels++; // 统计有效像素
            }
            else
            {
                Vec2f flow;
                flow[0] = std::numeric_limits<float>::quiet_NaN();
                flow[1] = std::numeric_limits<float>::quiet_NaN();
                gt.at<Vec2f>(v, u) = flow;
            }
        }
    }

    // 添加统计输出
    printf("Flow GT统计: %d/%d (%.2f%%) 有效像素\n",
           valid_pixels, width * height, 100.0f * valid_pixels / (width * height));

    return gt;
}

/**
 * @brief 计算每个像素的3D场景流端点误差（欧式距离）
 * @param pred_scene_flow  预测的场景流（CV_32FC3, 每像素(x, y, z)单位：米）
 * @param gt_scene_flow    Ground Truth 场景流（CV_32FC3, 单位：米，NaN为无效像素）
 * @return                 CV_32FC1，每像素的EPE误差（float，单位：米）
 *
 * 实现要点说明
1.功能
    计算每个像素的3D场景流欧式距离（End-Point Error, EPE），即预测与GT三维向量的L2范数。
2.输入参数
    pred_scene_flow：你的方法输出的场景流（每像素3D向量，单位米）。
    gt_scene_flow：GT场景流（单位米，NaN为无效像素）。
3.输出结果
    返回CV_32FC1的误差图，每个像素为float，表示该点的EPE，单位米。GT无效像素为NaN。
4.健壮性
    仅对GT为有效数的像素（即不是NaN）计算误差，防止无效像素污染统计。
    这样后续用eval_mask时可以直接掩盖。
5.用法
    可直接与generateEvaluationMask()的掩码联合统计平均EPE、R0.1等。
 */
static Mat sceneFlowEndpointError(const cv::Mat &pred_scene_flow, const cv::Mat &gt_scene_flow)
{
    CV_Assert(pred_scene_flow.type() == CV_32FC3 && gt_scene_flow.type() == CV_32FC3);
    CV_Assert(pred_scene_flow.size() == gt_scene_flow.size());

    cv::Mat epe_map(pred_scene_flow.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));

    for (int v = 0; v < pred_scene_flow.rows; ++v)
    {
        for (int u = 0; u < pred_scene_flow.cols; ++u)
        {
            cv::Vec3f pred = pred_scene_flow.at<cv::Vec3f>(v, u);
            cv::Vec3f gt = gt_scene_flow.at<cv::Vec3f>(v, u);

            // 判断GT有效（不是NaN），才统计误差
            if (cv::checkRange(gt))
            {
                float dx = pred[0] - gt[0];
                float dy = pred[1] - gt[1];
                float dz = pred[2] - gt[2];
                float epe = std::sqrt(dx * dx + dy * dy + dz * dz);
                epe_map.at<float>(v, u) = epe;
            }
            // 否则epe为NaN，后续mask评估时会被自动忽略
        }
    }
    return epe_map;
}

/**
 * @brief 根据场景流误差图生成彩色热力图（Heatmap），便于直观展示误差分布。
 *
 * 实现要点说明：
 * - 输入误差图（CV_32FC1，单位：米），仅对eval_mask=255的像素可视化，其他像素为黑色。
 * - 支持手动设置误差可视化上限（max_error，推荐0.3~1.0），便于高误差区域对比。
 * - 首先将误差归一化到[0,255]，再用OpenCV applyColorMap（如COLORMAP_JET）上色。
 * - 掩码内为有效像素着色，掩码外全部为黑色。
 * - 输出为CV_8UC3彩色BGR图，可直接用于保存或可视化。
 *
 * @param error_map    单通道误差图（CV_32FC1，单位：米）
 * @param eval_mask    单通道掩码（CV_8UC1，255=有效像素，0=无效），由generateEvaluationMask生成
 * @param max_error    可视化的误差最大值（如0.3米），超过的误差统一显示为红色
 * @return             CV_8UC3 彩色热图（BGR，掩码外为黑色）
 */
static Mat sceneFlowErrorHeatMap(const cv::Mat &error_map, const cv::Mat &eval_mask, float max_error = 0.3f)
{
    CV_Assert(error_map.type() == CV_32F && eval_mask.type() == CV_8U);
    CV_Assert(error_map.size() == eval_mask.size());

    // 1. 归一化误差（限定最大误差）
    cv::Mat normalized_error = cv::Mat::zeros(error_map.size(), CV_8U);
    for (int v = 0; v < error_map.rows; ++v)
    {
        for (int u = 0; u < error_map.cols; ++u)
        {
            if (eval_mask.at<uchar>(v, u) > 0)
            {
                float val = error_map.at<float>(v, u);
                if (!std::isfinite(val))
                    val = 0.0f;
                float norm_val = std::min(val, max_error) / max_error * 255.0f;
                normalized_error.at<uchar>(v, u) = static_cast<uchar>(norm_val);
            }
        }
    }

    // 2. 应用伪彩色映射（J=低误差蓝，高误差红）
    cv::Mat heatmap;
    cv::applyColorMap(normalized_error, heatmap, cv::COLORMAP_JET);

    // 3. 掩码外设为黑色（可选，强化可视化效果）
    cv::Mat masked_heatmap = heatmap.clone();
    for (int v = 0; v < eval_mask.rows; ++v)
    {
        for (int u = 0; u < eval_mask.cols; ++u)
        {
            if (eval_mask.at<uchar>(v, u) == 0)
                masked_heatmap.at<cv::Vec3b>(v, u) = cv::Vec3b(0, 0, 0);
        }
    }
    return masked_heatmap;
}

/**
 * @brief 计算场景流误差超过指定阈值的有效像素比例（如 KITTI SF-all，误差>0.1m 占比）
 *
 * 实现要点说明：
 * - 只在eval_mask=255的像素范围内进行统计（即仅统计有效且在指定区域的像素）。
 * - 支持任意阈值（如0.1m/0.2m），与KITTI官网SF-all等指标完全对应。
 * - 返回值为百分比（0~100），可直接用于评估表格输出。
 * - 无效像素（掩码外）全部跳过。
 *
 * @param error_map  单通道float误差图（CV_32FC1，单位：米）
 * @param eval_mask  单通道掩码（CV_8UC1，255=有效，0=无效），通常由generateEvaluationMask生成
 * @param threshold  判断阈值（如0.1，表示统计误差>0.1m的像素占比）
 * @return           float，百分比（0~100），即大误差像素所占比例
 */
static float sceneFlowStat_RX(const cv::Mat &error_map, const cv::Mat &eval_mask, float threshold)
{
    CV_Assert(error_map.type() == CV_32F && eval_mask.type() == CV_8U);
    CV_Assert(error_map.size() == eval_mask.size());

    int total_valid = 0;
    int above_thresh = 0;
    std::vector<float> all_errors; // 添加调试

    for (int v = 0; v < error_map.rows; ++v)
    {
        for (int u = 0; u < error_map.cols; ++u)
        {
            if (eval_mask.at<uchar>(v, u) > 0)
            {
                float val = error_map.at<float>(v, u);
                if (!std::isfinite(val))
                    continue; // 跳过NaN

                all_errors.push_back(val); // 调试用
                ++total_valid;
                if (val > threshold)
                    ++above_thresh;
            }
        }
    }

    // ⭐️ 添加调试信息
    if (!all_errors.empty())
    {
        std::sort(all_errors.begin(), all_errors.end());
        printf("R%.1f统计: %d/%d (%.2f%%) 误差>%.1fm, 误差范围[%.4f, %.4f]\n",
               threshold, above_thresh, total_valid,
               100.0f * above_thresh / total_valid, threshold,
               all_errors[0], all_errors.back());
    }

    if (total_valid == 0)
        return 0.0f;
    return 100.0f * above_thresh / total_valid; // 返回百分比
}

/**
 * @brief 计算场景流误差分布的A百分位（如A90，90%像素的误差不超过多少米）
 *
 * 实现要点说明：
 * - 仅对eval_mask=255的有效像素统计，掩码外全部跳过。
 * - 支持任意百分位，如0.9（A90）、0.95（A95）等，与KITTI/Sintel官网A90等指标一致。
 * - 返回误差值（float，单位米），即百分位像素的EPE阈值。
 * - 若无有效像素，返回0.0f。
 *
 * @param error_map   单通道误差图（CV_32FC1，单位：米）
 * @param eval_mask   单通道掩码（CV_8UC1，255=有效，0=无效），通常由generateEvaluationMask生成
 * @param percentile  百分位（如0.9表示A90）
 * @return            float，EPE误差阈值（米）
 */
static float sceneFlowStat_AX(const cv::Mat &error_map, const cv::Mat &eval_mask, float percentile)
{
    CV_Assert(error_map.type() == CV_32F && eval_mask.type() == CV_8U);
    CV_Assert(error_map.size() == eval_mask.size());
    CV_Assert(percentile > 0.0 && percentile <= 1.0);

    // 收集所有有效像素的误差值
    std::vector<float> valid_errors;
    for (int v = 0; v < error_map.rows; ++v)
    {
        for (int u = 0; u < error_map.cols; ++u)
        {
            // 仅统计eval_mask=255的有效像素
            // eval_mask.at<uchar>(v, u) > 0 表示该像素在评估区域内
            // error_map.at<float>(v, u) 获取该像素的误差值
            if (eval_mask.at<uchar>(v, u) > 0)
            {
                float err = error_map.at<float>(v, u);
                if (std::isfinite(err))
                    valid_errors.push_back(err);
            }
        }
    }

    if (valid_errors.empty())
    {
        printf("⚠️ A90计算：没有有效误差数据\n");
        return 0.0f;
    }

    // ⭐️ 添加调试信息
    printf("A90计算：收集到%zu个有效误差值，最小=%.4f，最大=%.4f\n",
           valid_errors.size(),
           *std::min_element(valid_errors.begin(), valid_errors.end()),
           *std::max_element(valid_errors.begin(), valid_errors.end()));

    // 计算百分位误差
    // 注意：percentile是0.0~1.0之间的值，表示百分比位置
    // 例如：0.9表示90%的像素误差不超过该值
    std::sort(valid_errors.begin(), valid_errors.end());
    size_t idx = static_cast<size_t>(percentile * valid_errors.size());
    if (idx >= valid_errors.size())
        idx = valid_errors.size() - 1;

    return valid_errors[idx]; // 返回百分位误差值（单位：米）
}

/**
 * @brief 严格KITTI官方标准：统计“EPE > 0.03m（3cm）且相对误差 > 5%”的错误像素比例
 * @param pred 预测场景流（CV_32FC3）
 * @param gt GT场景流（CV_32FC3）
 * @param mask 有效区域mask（CV_8U，非零为有效像素）
 * @return 错误像素占比（百分比，0~100%）
 */
static float sceneFlowStat_OfficialError(const cv::Mat &pred, const cv::Mat &gt, const cv::Mat &mask)
{
    int error_count = 0;
    int valid_count = 0;
    for (int v = 0; v < pred.rows; ++v)
    {
        for (int u = 0; u < pred.cols; ++u)
        {
            if (mask.empty() || mask.at<uchar>(v, u) > 0)
            {
                cv::Vec3f gt_vec = gt.at<cv::Vec3f>(v, u);
                cv::Vec3f pred_vec = pred.at<cv::Vec3f>(v, u);

                // ⭐️ 添加有效性检查
                if (!cv::checkRange(gt_vec) || !cv::checkRange(pred_vec))
                {
                    continue; // 跳过无效像素
                }

                float abs_err = cv::norm(pred_vec - gt_vec);
                float gt_norm = cv::norm(gt_vec);

                // ⭐️ 添加分母检查
                if (gt_norm < 1e-6f)
                {
                    continue; // 跳过零向量
                }

                float rel_err = abs_err / gt_norm;

                if (abs_err > 0.03f && rel_err > 0.05f)
                {
                    error_count++;
                }
                valid_count++;
            }
        }
    }
    if (valid_count == 0)
        return 0.0f;
    return 100.0f * error_count / valid_count;
}

/**
 * @brief 计算一帧场景流的主要误差指标（均值EPE，R0.1，R0.2，A90）并输出、可选可视化
 *
 * 实现要点说明：
 * - 输入预测场景流与GT、评估mask，可自动调用endpointError、Stat_RX/AX等函数
 * - 自动输出所有指标（可打印、可保存）
 * - 可选生成/显示误差热图
 * - 建议输出vector<double>或结构体用于后续写CSV
 *
 * @param pred_scene_flow 预测场景流（CV_32FC3，单位米）
 * @param gt_scene_flow   GT场景流（CV_32FC3，单位米，NaN为无效）
 * @param image          原始输入图像（BGR/灰度，用于生成区域mask/可选）
 * @param region         区域类型（"all"/"discontinuities"/"untextured"）
 * @param display_images 是否显示可视化窗口（热图/方向图）
 * @param stats_vector   存储结果的vector（输出参数）
 * @return 无（结果通过stats_vector返回，并可显示/保存）
 */
void EvaluateSceneFlow::calculateSceneFlowStats(const cv::Mat &pred_scene_flow,
                                                const cv::Mat &gt_scene_flow,
                                                const cv::Mat &image,
                                                const std::string &region,
                                                bool display_images,
                                                std::vector<double> &stats_vector)
{
    // === 1. 生成评估掩码 ===
    cv::Mat eval_mask = generateEvaluationMask(gt_scene_flow, image, region);

    // === 2. 计算端点误差图（EPE） ===
    cv::Mat epe_map = sceneFlowEndpointError(pred_scene_flow, gt_scene_flow);

    // === 3. 主要统计指标 ===
    // (1) 平均EPE
    double meanEPE = cv::mean(epe_map, eval_mask)[0];

    // (2) R0.1, R0.2（大误差占比，百分比）
    float R01 = sceneFlowStat_RX(epe_map, eval_mask, 0.1f); // >0.1m
    float R02 = sceneFlowStat_RX(epe_map, eval_mask, 0.2f); // >0.2m

    // (3) A90（90%像素的EPE下限）
    float A90 = sceneFlowStat_AX(epe_map, eval_mask, 0.9f);
    float official_err = sceneFlowStat_OfficialError(pred_scene_flow, gt_scene_flow, eval_mask);

    printf("KITTI官方 SF错误率: %.2f%%\n", official_err);

    // 可选：有效像素总数
    int valid_count = cv::countNonZero(eval_mask);

    // === 4. 存入vector，建议顺序：meanEPE, R0.1, R0.2, A90, official_err, valid_count ===
    stats_vector.clear();
    stats_vector.push_back(meanEPE);
    stats_vector.push_back(R01);
    stats_vector.push_back(R02);
    stats_vector.push_back(A90);
    stats_vector.push_back(official_err);
    stats_vector.push_back(valid_count);

    // // === 5. 可选可视化 ===
    // if (display_images)
    // {
    //     // 误差热图
    //     cv::Mat heatmap = sceneFlowErrorHeatMap(epe_map, eval_mask, 0.3f);
    //     cv::imshow("Scene Flow Error Heatmap", heatmap);

    //     // 方向/幅值彩图
    //     cv::Mat vis = sceneFlowToDisplay(pred_scene_flow, eval_mask, 2.0f);
    //     cv::imshow("Scene Flow Direction Visualization", vis);

    //     cv::waitKey(10);
    // }

    // === 6. 输出控制台打印 ===
    printf("【%s | %s】有效像素: %d  Mean EPE: %.4f  SF Err: %.2f%%  R0.1: %.2f%%  R0.2: %.2f%%  A90: %.4fm\n",
           region.c_str(), "SceneFlow", valid_count, meanEPE, official_err, R01, R02, A90);
}

/**
 * @brief 将一帧的场景流评估统计信息追加写入CSV文件（自动新建/追加）
 *
 * 实现要点说明：
 * - 支持自动新建csv并写表头，之后自动追加，不重复表头。
 * - 每次写入一行：image_no, method, meanEPE, R0.1, R0.2, A90, 时间等。
 * - 支持多线程/多帧批量评估时重复调用，不丢数据。
 * - 指标可用std::vector<double>或struct传入。
 *
 * @param stats_vector  存储本帧评估指标的vector（推荐顺序：image_no, meanEPE, R0.1, R0.2, A90, time(ms)）
 * @param method        光流/场景流算法名称
 * @param image_no      当前图像编号
 * @param csv_path      CSV文件路径（如"../data/outputs/scene_flow_stats.csv"）
 * @param write_header  是否自动写表头（首次或外部判定时传true）
 * @return              返回true表示写入成功
 */
static bool writeSceneFlowStatsToCSV(const std::vector<double> &stats_vector,
                                     const std::string &method,
                                     int image_no,
                                     const std::string &csv_path,
                                     bool write_header = false,
                                     double time_ms = 0.0)
{
    std::ofstream file;
    // 用追加模式写入，首帧时write_header写表头
    file.open(csv_path, std::ios::out | std::ios::app);
    if (!file.is_open())
    {
        printf("❌ Failed to open CSV file: %s\n", csv_path.c_str());
        return false;
    }

    // 判断文件是否为空（或者外部首帧传write_header=true）
    // static bool header_written = false;
    if (write_header)
    {
        file << "image_no,method,meanEPE,R0.1(%),R0.2(%),A90(m),official_error(%),valid_count,time_ms\n";
        // header_written = true;
    }

    // 写一行数据
    file << image_no << ",";
    file << method << ",";
    for (size_t i = 0; i < stats_vector.size(); ++i)
    {
        file << stats_vector[i];
        file << ",";
    }
    file << time_ms << "\n"; // 在最后加上时间
    file << "\n";

    file.close();
    return true;
}

/**
 * @brief 工程级终极版：KITTI场景流评估主入口
 *        多区域(mask/统计/写csv)，主区域保存可视化图，支持多算法可扩展
 *
 * @param method           光流法名称（如"degraf_flow_cudalk"）
 * @param display_images   是否弹窗显示主观可视化
 * @param image_no         测试帧编号
 * @return                 0=成功, -1=失败
 */
int EvaluateSceneFlow::runEvaluation(const std::string &method, bool display_images, int image_no)
{

    cv::Mat img = cv::imread("/app/data/kitti_sceneflow/data_scene_flow/training/image_2/000000_10.png", cv::IMREAD_UNCHANGED);
    if (img.empty())
        std::cout << "OpenCV imread failed!" << std::endl;
    else
        std::cout << "OpenCV imread success! shape=" << img.rows << "x" << img.cols << " channels=" << img.channels() << std::endl;
    // [1] 路径构建
    char num[7];
    sprintf(num, "%06d", image_no);
    std::string num_str(num);
    std::string base_dir = "/app/data/kitti_sceneflow/data_scene_flow/training/";
    std::string i1_path = base_dir + "image_2/" + num_str + "_10.png";
    std::string i2_path = base_dir + "image_2/" + num_str + "_11.png";
    std::string disp0_path = base_dir + "disp_noc_0/" + num_str + "_10.png";
    std::string disp1_path = base_dir + "disp_noc_1/" + num_str + "_10.png";
    std::string calib_path = "/app/data/kitti_sceneflow/data_scene_flow_calib/training/calib_cam_to_cam/" + num_str + ".txt";
    std::string flow_gt_path = base_dir + "flow_noc/" + num_str + "_10.png";
    std::string output_dir = "../data/outputs/";

    // [2] 加载输入与预处理
    cv::Mat i1 = cv::imread(i1_path, 1);
    cv::Mat i2 = cv::imread(i2_path, 1);
    cv::Mat disp0 = cv::imread(disp0_path, cv::IMREAD_UNCHANGED);
    cv::Mat disp1 = cv::imread(disp1_path, cv::IMREAD_UNCHANGED);
    cv::Mat flow_gt = cv::imread(flow_gt_path, cv::IMREAD_UNCHANGED);

    if (i1.empty() || i2.empty() || disp0.empty() || disp1.empty() || flow_gt.empty())
    {
        printf("❌ 输入缺失 %06d\n", image_no);
        return -1;
    }
    if (i1.size() != i2.size() || i1.size() != disp0.size() || disp0.size() != disp1.size() || i1.size() != flow_gt.size())
    {
        printf("❌ 尺寸不一致 %06d\n", image_no);
        return -1;
    }

    // [3] 光流估计
    cv::Mat gray1, gray2;
    if (i1.channels() == 3)
        cv::cvtColor(i1, gray1, cv::COLOR_BGR2GRAY);
    else
        gray1 = i1.clone();
    if (i2.channels() == 3)
        cv::cvtColor(i2, gray2, cv::COLOR_BGR2GRAY);
    else
        gray2 = i2.clone();
    cv::Mat flow;
    double t_start = cv::getTickCount();
    if (method == "farneback")
        cv::optflow::createOptFlow_Farneback()->calc(gray1, gray2, flow);
    else if (method == "tvl1")
        cv::optflow::createOptFlow_DualTVL1()->calc(gray1, gray2, flow);
    else if (method == "deepflow")
        cv::optflow::createOptFlow_DeepFlow()->calc(gray1, gray2, flow);
    else if (method == "DISflow_fast")
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST)->calc(gray1, gray2, flow);
    else if (method == "DISflow_medium")
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM)->calc(gray1, gray2, flow);
    else if (method == "degraf_flow_rlof")
    {
        FeatureMatcher matcher;
        matcher.degraf_flow_RLOF(i1, i2, flow, 127, 0.05f, true, 500.0f, 1.5f);
    }
    // else if (method == "degraf_flow_cudalk")
    // {
    //     FeatureMatcher matcher;
    //     matcher.degraf_flow_CudaLK(i1, i2, flow, 127, 0.05f, true, 500.0f, 1.5f);
    // }
    // else if (method == "degraf_flow_gpu")
    // {
    //     FeatureMatcher matcher;
    //     matcher.degraf_flow_GPU(i1, i2, flow, 8, 0.01f, true, 500.0f, 1.5f);
    // }
    else
    {
        printf("❌ 未知光流法: %s\n", method.c_str());
        return -1;
    }
    double time_ms = (cv::getTickCount() - t_start) / cv::getTickFrequency() * 1000.0;
    if (flow.empty())
    {
        printf("❌ 光流计算失败 %06d\n", image_no);
        return -1;
    }

    // [4] 场景流重建
    float fx, fy, cx, cy, baseline;
    if (!loadCameraIntrinsics(calib_path, fx, fy, cx, cy, baseline))
    {
        fx = 721.5377f;
        fy = 721.5377f;
        cx = 609.5593f;
        cy = 172.8540f;
        baseline = 0.5371f;
        printf("⚠️ Using default KITTI camera parameters\n");
    }
    // cv::Mat disp0_f32, disp1_f32;
    // disp0.convertTo(disp0_f32, CV_32F, (disp0.type() == CV_16UC1) ? 1.0 / 256.0 : 1.0);
    // disp1.convertTo(disp1_f32, CV_32F, (disp1.type() == CV_16UC1) ? 1.0 / 256.0 : 1.0);
    cv::Mat disp0_f32 = readKITTIDisparity(disp0_path);
    cv::Mat disp1_f32 = readKITTIDisparity(disp1_path);
    SceneFlowReconstructor reconstructor(fx, fy, cx, cy, baseline);
    cv::Mat scene_flow = reconstructor.computeSceneFlow(flow, disp0_f32, disp1_f32);
    if (scene_flow.empty())
    {
        printf("❌ 场景流计算失败 %06d\n", image_no);
        return -1;
    }
    // 读取光流GT
    cv::Mat flow_gt_processed = readKITTIFlowGT(flow_gt_path);
    // [5] 读取GT Scene Flow
    cv::Mat gt_scene_flow = reconstructor.computeSceneFlow(flow_gt_processed, disp0_f32, disp1_f32);
    if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3)
    {
        printf("❌ GT场景流读取失败 %06d\n", image_no);
        return -1;
    }
    // if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3)
    // {
    //     std::cout << "❌ Fake GT for debug: using zeros for " << objmap_path << std::endl;
    //     int height = disp0.rows;
    //     int width = disp0.cols;
    //     gt_scene_flow = cv::Mat::zeros(height, width, CV_32FC3);
    //     // printf("❌ GT场景流缺失 %06d\n", image_no);
    //     // return -1;
    // }

    // [6] 多区域评估、写CSV，只为主区域可视化
    std::vector<std::string> regions = {"all", "discontinuities", "untextured"};
    std::vector<double> stats_vector;
    std::string csv_path = output_dir + "scene_flow_stats.csv";
    bool header_written = false;
    for (const auto &region : regions)
    {
        // 6.1 评估统计
        stats_vector.clear(); // 清空上次统计结果
        calculateSceneFlowStats(scene_flow, gt_scene_flow, i1, region, false, stats_vector);
        // 6.2 写CSV - 只在第一个区域且第一帧时写表头，其他时候追加
        bool write_header = (image_no == 0 && !header_written && region == "all");
        if (write_header)
        {
            // 清空文件并写新表头
            std::ofstream clear_file(csv_path, std::ios::out | std::ios::trunc);
            clear_file.close();
        }
        writeSceneFlowStatsToCSV(stats_vector, method + "_" + region, image_no, csv_path, write_header, time_ms);
        if (write_header)
            header_written = true;

        // 6.3 可视化（只对主区域all保存图片/弹窗）
        if (display_images && region == "all")
        {
            cv::Mat eval_mask = generateEvaluationMask(gt_scene_flow, i1, region);

            // 1. 场景流方向可视化图
            cv::Mat flow_vis = sceneFlowToDisplay(scene_flow, eval_mask);

            // 2. 场景流误差热图
            cv::Mat epe_map = sceneFlowEndpointError(scene_flow, gt_scene_flow);
            cv::Mat heatmap = sceneFlowErrorHeatMap(epe_map, eval_mask);

            // 3. 原图BGR加标题
            cv::Mat img_label = i1.clone();
            cv::putText(img_label, "Image", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            // 4. 方向图加标题
            cv::Mat flow_label = flow_vis.clone();
            cv::putText(flow_label, "SceneFlow Direction", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            // 5. 热图加标题
            cv::Mat heatmap_label = heatmap.clone();
            cv::putText(heatmap_label, "SceneFlow Error", cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            // 6. 保证尺寸一致
            cv::resize(flow_label, flow_label, img_label.size());
            cv::resize(heatmap_label, heatmap_label, img_label.size());

            // === ⭐️ 单独保存每张小图 ===
            cv::imwrite(output_dir + num + "_" + method + "_img.png", img_label);
            cv::imwrite(output_dir + num + "_" + method + "_flow_vis.png", flow_label);
            cv::imwrite(output_dir + num + "_" + method + "_heatmap.png", heatmap_label);

            // === ⭐️ 拼接大图 ===
            std::vector<cv::Mat> rows = {img_label, flow_label, heatmap_label};
            cv::Mat combined;
            cv::vconcat(rows, combined);

            cv::imwrite(output_dir + num + "_" + method + "_allvis.png", combined);
            // cv::imshow("SceneFlow Results", combined);
            // cv::waitKey(0); // 按任意键关闭
        }
    }

    printf("✅ Scene flow evaluation finished for %06d\n", image_no);
    return 0;
}