#include "EvaluateSceneFlow.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/image_viewer.h>
// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/common/time.h>

using namespace cv;
using namespace std;

EvaluateSceneFlow::EvaluateSceneFlow() {}

/**
 * Function: Parse fx, fy, cx, cy, baseline from KITTI calib_cam_to_cam/*.txt files
 * Input parameters:
 * calib_file: file path
 * fx, fy, cx, cy, baseline: reference variables, write parsed values
 * Output: true means successful parsing, false means failure or incorrect format
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
            std::istringstream iss(line.substr(11)); // Skip label
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

    // From P2 parse intrinsic parameters
    fx = P2_values[0]; // P2[0][0]
    fy = P2_values[5]; // P2[1][1]
    cx = P2_values[2]; // P2[0][2]
    cy = P2_values[6]; // P2[1][2]

    // From P2[0][3] 和 P3[0][3] calculate baseline
    float Tx2 = P2_values[3]; // P2[0][3]
    float Tx3 = P3_values[3]; // P3[0][3]
    baseline = -(Tx3 - Tx2) / fx;

    return true;
}

/**
 * @brief Read disparity GT file according to KITTI official standard
 * @param path disparity file path (such as disp_noc_0/000000_10.png)
 * @return CV_32FC1 disparity map, invalid pixels are NaN
 */
static cv::Mat readKITTIDisparity(const std::string &path)
{
    // 1. Read the original uint16 image
    cv::Mat disp_raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (disp_raw.empty())
    {
        printf("❌ Cannot open disparity file: %s\n", path.c_str());
        return cv::Mat();
    }

    // 2. Check type: KITTI disparity should be CV_16UC1
    if (disp_raw.type() != CV_16UC1)
    {
        printf("❌ Invalid disparity image type (expect CV_16UC1, got %d)\n", disp_raw.type());
        return cv::Mat();
    }

    // 3. Convert according to KITTI standard
    cv::Mat disp_f32(disp_raw.size(), CV_32F);
    for (int y = 0; y < disp_raw.rows; ++y)
    {
        for (int x = 0; x < disp_raw.cols; ++x)
        {
            uint16_t raw_val = disp_raw.at<uint16_t>(y, x);
            if (raw_val == 0)
            {
                //  KITTI standard: 0 value indicates invalid pixel
                disp_f32.at<float>(y, x) = std::numeric_limits<float>::quiet_NaN();
            }
            else
            {
                //  KITTI standard: divide by 256 to get true disparity value
                disp_f32.at<float>(y, x) = static_cast<float>(raw_val) / 256.0f;
            }
        }
    }
    return disp_f32;
}

/**
 * @brief Read optical flow GT file according to KITTI official standard
 * @param path Optical flow file path (such as flow_noc/000000_10.png or flow_occ/000000_10.png)
 * @return CV_32FC2 optical flow map, invalid pixels are NaN
 */
static cv::Mat readKITTIFlowGT(const std::string &ground_truth_path)
{
    String path = ground_truth_path;
    // NB opencv has order BGR => valid , v , u
    Mat image = imread(path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    Mat gt = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
    int width = image.cols;
    int height = image.rows;
    int valid_pixels = 0; // Add statistics

    for (int32_t v = 0; v < height; v++)
    {
        for (int32_t u = 0; u < width; u++)
        {
            Vec3s val = image.at<Vec3s>(v, u);
            if (val[0] > 0) // validity check
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
                valid_pixels++; // Count valid pixels
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

    // Add statistics output
    printf("Flow GT statistics: %d/%d (%.2f%%) valid pixels\n",
           valid_pixels, width * height, 100.0f * valid_pixels / (width * height));

    return gt;
}

// 🆕 新增：相对误差计算
static cv::Mat calculateRelativeError(const cv::Mat &epe_map, const cv::Mat &gt_scene_flow)
{
    cv::Mat rel_error = cv::Mat::zeros(epe_map.size(), CV_32F);

    for (int v = 0; v < epe_map.rows; ++v)
    {
        for (int u = 0; u < epe_map.cols; ++u)
        {
            cv::Vec3f gt_vec = gt_scene_flow.at<cv::Vec3f>(v, u);
            float gt_norm = cv::norm(gt_vec);
            float epe = epe_map.at<float>(v, u);

            if (gt_norm > 1e-6f && !std::isnan(epe))
            {
                rel_error.at<float>(v, u) = epe / gt_norm;
            }
            else
            {
                rel_error.at<float>(v, u) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return rel_error;
}

// 🆕 新增：标准指标计算（参考学术标准）
SceneFlowMetrics EvaluateSceneFlow::calculateStandardMetrics(const cv::Mat &pred_scene_flow,
                                                             const cv::Mat &gt_scene_flow)
{
    SceneFlowMetrics metrics;

    int total_gt_valid = 0;
    int evaluable_points = 0;

    // 🆕 添加失效区域分析变量
    std::vector<float> missing_region_magnitudes;
    std::vector<float> covered_region_magnitudes;

    double total_epe = 0.0;
    int acc_strict_count = 0;
    int acc_relax_count = 0;
    int outlier_count = 0;

    for (int v = 0; v < pred_scene_flow.rows; ++v)
    {
        for (int u = 0; u < pred_scene_flow.cols; ++u)
        {
            cv::Vec3f pred = pred_scene_flow.at<cv::Vec3f>(v, u);
            cv::Vec3f gt = gt_scene_flow.at<cv::Vec3f>(v, u);

            // 检查GT有效性
            bool gt_valid = (!std::isnan(gt[0]) && !std::isnan(gt[1]) && !std::isnan(gt[2]) &&
                             !(gt[0] == 0 && gt[1] == 0 && gt[2] == 0));

            if (gt_valid)
            {
                total_gt_valid++;

                // 🆕 计算GT幅值
                float gt_magnitude = cv::norm(gt);

                // 检查pred有效性
                bool pred_valid = (!std::isnan(pred[0]) && !std::isnan(pred[1]) && !std::isnan(pred[2]) &&
                                   !(pred[0] == 0 && pred[1] == 0 && pred[2] == 0));

                if (pred_valid)
                {
                    evaluable_points++;
                    covered_region_magnitudes.push_back(gt_magnitude); // 记录覆盖区域幅值

                    float dx = pred[0] - gt[0];
                    float dy = pred[1] - gt[1];
                    float dz = pred[2] - gt[2];
                    float epe = std::sqrt(dx * dx + dy * dy + dz * dz);

                    total_epe += epe;

                    float gt_norm = cv::norm(gt);
                    float rel_err = epe / (gt_norm + 1e-6f);

                    if (epe < 0.05f || rel_err < 0.05f)
                        acc_strict_count++;
                    if (epe < 0.1f || rel_err < 0.1f)
                        acc_relax_count++;
                    if (epe > 0.3f || rel_err > 0.1f)
                        outlier_count++;
                }
                else
                {
                    missing_region_magnitudes.push_back(gt_magnitude); // 记录失效区域幅值
                }
            }
        }
    }

    // 失效区域分析
    if (!missing_region_magnitudes.empty() && !covered_region_magnitudes.empty())
    {
        float avg_missing = std::accumulate(missing_region_magnitudes.begin(),
                                            missing_region_magnitudes.end(), 0.0f) /
                            missing_region_magnitudes.size();
        float avg_covered = std::accumulate(covered_region_magnitudes.begin(),
                                            covered_region_magnitudes.end(), 0.0f) /
                            covered_region_magnitudes.size();

        printf("=== 区域分析 ===\n");
        printf("覆盖区域: %zu像素, 平均GT幅值: %.4f m\n",
               covered_region_magnitudes.size(), avg_covered);
        printf("失效区域: %zu像素, 平均GT幅值: %.4f m\n",
               missing_region_magnitudes.size(), avg_missing);

        if (avg_missing > avg_covered)
        {
            printf("⚠️  失效区域比覆盖区域更困难 (大运动)\n");
        }
        else
        {
            printf("✅ 失效区域比覆盖区域更简单 (小运动)\n");
        }
        printf("==================\n");
    }

    if (evaluable_points > 0)
    {
        metrics.EPE3d = total_epe / evaluable_points;
        metrics.AccS = 100.0 * acc_strict_count / evaluable_points;
        metrics.AccR = 100.0 * acc_relax_count / evaluable_points;
        metrics.Outlier = 100.0 * outlier_count / evaluable_points;
        metrics.valid_count = evaluable_points;
    }

    return metrics;
}

// 🆕 新增：核心单帧评估函数
SceneFlowMetrics EvaluateSceneFlow::evaluateSingleFrame(const cv::Mat &pred_scene_flow,
                                                        const cv::Mat &gt_scene_flow,
                                                        bool verbose)
{
    SceneFlowMetrics metrics = calculateStandardMetrics(pred_scene_flow, gt_scene_flow);

    if (verbose)
    {
        printf("EPE3d: %.4f | AccS: %.2f%% | AccR: %.2f%% | Outlier: %.2f%% | Valid: %d\n",
               metrics.EPE3d, metrics.AccS, metrics.AccR, metrics.Outlier, metrics.valid_count);
    }

    return metrics;
}

// 🔧 修改：简化CSV写入
void EvaluateSceneFlow::writeMetricsToCSV(const SceneFlowMetrics &metrics,
                                          const std::string &method,
                                          int image_no,
                                          const std::string &csv_path)
{
    std::ofstream file(csv_path, std::ios::app);
    if (!file.is_open())
    {
        printf("❌ Failed to open CSV: %s\n", csv_path.c_str());
        return;
    }

    // 写入标准格式
    file << image_no << "," << method << ","
         << metrics.EPE3d << "," << metrics.AccS << "," << metrics.AccR << ","
         << metrics.Outlier << "," << metrics.valid_count << "," << metrics.time_ms << "\n";

    file.close();
}

/**
 * @brief Engineering-level ultimate version: KITTI scene flow evaluation main entrance
 * Multi-area (mask/statistics/write csv), the main area saves the visualization, supports multiple algorithms and is scalable
 *
 * @param method Optical flow method name (such as "degraf_flow_cudalk")
 * @param display_images Whether to display subjective visualization in a pop-up window
 * @param image_no Test frame number
 * @return 0=success, -1=failure
 */
SceneFlowMetrics EvaluateSceneFlow::runEvaluation(const std::string &method, bool display_images, int image_no)
{

    // [0] create initial error result
    SceneFlowMetrics error_result;
    // [1] Path construction
    char num[7];
    sprintf(num, "%06d", image_no);
    std::string num_str(num);
    std::string base_dir = "../data/data_scene_flow/training/";
    std::string i1_path = base_dir + "image_2/" + num_str + "_10.png";
    std::string i2_path = base_dir + "image_2/" + num_str + "_11.png";
    std::string disp0_path = base_dir + "disp_noc_0/" + num_str + "_10.png";
    std::string disp1_path = base_dir + "disp_noc_1/" + num_str + "_10.png";
    std::string calib_path = "../data/data_scene_flow_calib/training/calib_cam_to_cam/" + num_str + ".txt";
    std::string flow_gt_path = base_dir + "flow_noc/" + num_str + "_10.png";
    std::string output_dir = "../data/outputs/";

    // [2] Loading input and preprocessing
    cv::Mat i1 = cv::imread(i1_path, 1);
    cv::Mat i2 = cv::imread(i2_path, 1);
    cv::Mat disp0 = cv::imread(disp0_path, cv::IMREAD_UNCHANGED);
    cv::Mat disp1 = cv::imread(disp1_path, cv::IMREAD_UNCHANGED);
    cv::Mat flow_gt = cv::imread(flow_gt_path, cv::IMREAD_UNCHANGED);

    if (i1.empty() || i2.empty() || disp0.empty() || disp1.empty() || flow_gt.empty())
    {
        printf("❌ Input missing %06d\n", image_no);
        return error_result;
    }
    if (i1.size() != i2.size() || i1.size() != disp0.size() || disp0.size() != disp1.size() || i1.size() != flow_gt.size())
    {
        printf("❌ Size mismatch %06d\n", image_no);
        return error_result;
    }

    // [3] Optical flow estimation
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
    double flow_start = cv::getTickCount();
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
        matcher.degraf_flow_RLOF(i1, i2, flow, 127, 0.05f, true, 500.0f, 1.5f,num_str);
    }
    else if (method == "degraf_flow_lk")
    {
        FeatureMatcher matcher;
        matcher.degraf_flow_LK(i1, i2, flow, 127, 0.05f, true, 500.0f, 1.5f,num_str);
    }
    else if (method == "degraf_flow_cudalk")
    {
        FeatureMatcher matcher;
        matcher.degraf_flow_CudaLK(i1, i2, flow, 8, 0.01f, true, 500.0f, 1.5f,num_str);
    }
    else if (method == "degraf_flow_interponet")
    {
        FeatureMatcher matcher;
        matcher.degraf_flow_InterpoNet(i1, i2, flow, num_str);
    }
    else
    {
        printf("❌ Unknown optical flow method: %s\n", method.c_str());
        return error_result;
    }

    if (flow.empty())
    {
        printf("❌ Optical flow calculation failed %06d\n", image_no);
        return error_result;
    }

    // [4] Scene flow reconstruction
    float fx, fy, cx, cy, baseline;
    if (!loadCameraIntrinsics(calib_path, fx, fy, cx, cy, baseline))
    {
        fx = 721.5377f;
        fy = 721.5377f;
        cx = 609.5593f;
        cy = 172.8540f;
        baseline = 0.5371f;
        printf("Using default KITTI camera parameters\n");
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
        printf("Scene flow calculation failed %06d\n", image_no);
        return error_result; // Return early if scene flow is empty
    }

    // Read optical flow GT
    cv::Mat flow_gt_processed = readKITTIFlowGT(flow_gt_path);
    // [5] Read GT Scene Flow
    cv::Mat gt_scene_flow = reconstructor.computeSceneFlow(flow_gt_processed, disp0_f32, disp1_f32);
    if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3)
    {
        printf("GT Scene Flow reading failed %06d\n", image_no);
        return error_result;
    }
    double time_ms = (cv::getTickCount() - flow_start) / cv::getTickFrequency() * 1000.0;
    // if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3)
    // {
    //     std::cout << "Fake GT for debug: using zeros for " << objmap_path << std::endl;
    //     int height = disp0.rows;
    //     int width = disp0.cols;
    //     gt_scene_flow = cv::Mat::zeros(height, width, CV_32FC3);
    //     // printf("GTScene flow is missing %06d\n", image_no);
    //     // return -1;
    // }

    // [5] 🔧 简化：删除多区域循环，直接评估
    double t_start = cv::getTickCount();
    SceneFlowMetrics metrics = evaluateSingleFrame(scene_flow, gt_scene_flow, true);
    metrics.time_ms = time_ms;

    // [6] 🔧 简化：CSV写入
    std::string csv_path = "../data/outputs/scene_flow_results.csv";

    // 首帧写入表头
    if (image_no == 0)
    {
        std::ofstream header_file(csv_path, std::ios::trunc);
        header_file << "image_no,method,EPE3d,AccS(%),AccR(%),Outlier(%),valid_count,time_ms\n";
        header_file.close();
    }

    writeMetricsToCSV(metrics, method, image_no, csv_path);

    // [7] 保留：可选可视化（如果需要）
    // if (display_images)
    // {
    //     // 简化的可视化逻辑...
    // }

    // 存储结果用于批量统计
    all_results_.push_back(metrics);

    printf("✅ Frame %06d evaluated successfully\n", image_no);
    return metrics; // 返回评估结果对象
}