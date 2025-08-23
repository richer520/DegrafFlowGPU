#include "EvaluateSceneFlow.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>

using namespace cv;
using namespace std;

EvaluateSceneFlow::EvaluateSceneFlow() {}

// ===========================================
// 在文件开头添加KITTI颜色映射表
// ===========================================
// KITTI官方对数误差颜色映射表
static float LC[10][5] = {
    {0,0.0625,49,54,149},
    {0.0625,0.125,69,117,180},
    {0.125,0.25,116,173,209},
    {0.25,0.5,171,217,233},
    {0.5,1,224,243,248},
    {1,2,254,224,144},
    {2,4,253,174,97},
    {4,8,244,109,67},
    {8,16,215,48,39},
    {16,1000000000.0,165,0,38}
};



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

// Converts middlebury 2 channel flow image to 3 channel (CV_16UC3) KITTI format
/*!
\param flow 2 channel optical flow Mat, middlebury format
\return 3 channel KITTI optical flow Mat
*/
static Mat convertToKittiFlow(const Mat_<Point2f> &flow)
{

	Mat kittiFlow = cv::Mat::ones(flow.rows, flow.cols, CV_16UC3); // type 18

	int width = flow.cols;
	int height = flow.rows;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{

			const Point2f pt = flow(i, j);
			float u = pt.x;
			float v = pt.y;

			kittiFlow.at<Vec3s>(i, j)[1] = (uint16_t)max(min(v * 64.0f + 32768.0f, 65535.0f), 0.0f);

			kittiFlow.at<Vec3s>(i, j)[2] = (uint16_t)max(min(u * 64.0f + 32768.0f, 65535.0f), 0.0f);
		}
	}
	return kittiFlow;
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

// ===========================================
// 修改：generateSceneFlow4PanelVisualization() - 移除插值
// ===========================================
void EvaluateSceneFlow::generateSceneFlow4PanelVisualization(
    const std::string& method_name,
    int image_no,
    const cv::Mat& original_image,
    const cv::Mat& pred_scene_flow,
    const cv::Mat& gt_scene_flow,
    const std::string& output_path)
{
    int height = original_image.rows;
    int width = original_image.cols;
    
    // A) 原始图像
    cv::Mat panel_A = original_image.clone();
    
    // B) GT场景流 - 直接可视化，无插值
    cv::Mat panel_B = sceneFlowToColorMap(gt_scene_flow);
    
    // C) 预测场景流 - 直接可视化，无插值
    cv::Mat panel_C = sceneFlowToColorMap(pred_scene_flow);
    
    // D) 误差热图 - 基于原始数据
    cv::Mat panel_D = computeSceneFlowErrorMap(pred_scene_flow, gt_scene_flow);
    
    // 创建2x2布局
    cv::Mat result(height * 2, width * 2, CV_8UC3);
    panel_A.copyTo(result(cv::Rect(0, 0, width, height)));
    panel_B.copyTo(result(cv::Rect(width, 0, width, height)));
    panel_C.copyTo(result(cv::Rect(0, height, width, height)));
    panel_D.copyTo(result(cv::Rect(width, height, width, height)));
    
    cv::imwrite(output_path, result);
}

// ===========================================
// 修改：sceneFlowToColorMap() - 采用光流策略
// ===========================================
cv::Mat EvaluateSceneFlow::sceneFlowToColorMap(const cv::Mat& scene_flow_3d) {
    cv::Mat sf_copy = scene_flow_3d.clone();
    
    // 计算有效像素的统计信息
    std::vector<cv::Vec3f> valid_flows;
    for (int y = 0; y < sf_copy.rows; ++y) {
        for (int x = 0; x < sf_copy.cols; ++x) {
            cv::Vec3f sf = sf_copy.at<cv::Vec3f>(y, x);
            if (!std::isnan(sf[0]) && !std::isnan(sf[1]) && !std::isnan(sf[2]) && 
                (fabs(sf[0]) > 0.001f || fabs(sf[1]) > 0.001f)) {
                valid_flows.push_back(sf);
            }
        }
    }
    
    // 计算背景流（中位数）
    cv::Vec3f background_flow(0.0f, 0.0f, 0.0f);
    if (!valid_flows.empty()) {
        std::vector<float> x_vals, y_vals;
        for (const auto& flow : valid_flows) {
            x_vals.push_back(flow[0]);
            y_vals.push_back(flow[1]);
        }
        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        size_t mid = valid_flows.size() / 2;
        background_flow = cv::Vec3f(x_vals[mid], y_vals[mid], 0.0f);
    }
    
    // 关键：为所有无效像素填充背景流
    for (int y = 0; y < sf_copy.rows; ++y) {
        for (int x = 0; x < sf_copy.cols; ++x) {
            cv::Vec3f sf = sf_copy.at<cv::Vec3f>(y, x);
            // 如果是无效像素（NaN或接近零）
            if (std::isnan(sf[0]) || std::isnan(sf[1]) || std::isnan(sf[2]) ||
                (fabs(sf[0]) < 0.001f && fabs(sf[1]) < 0.001f && fabs(sf[2]) < 0.001f)) {
                sf_copy.at<cv::Vec3f>(y, x) = background_flow;
            }
        }
    }
    
    // 现在所有像素都有值，进行HSV映射
    cv::Mat color_map(sf_copy.size(), CV_8UC3);
    
    // 计算归一化参数
    float max_flow = 0.1f;
    for (int y = 0; y < sf_copy.rows; ++y) {
        for (int x = 0; x < sf_copy.cols; ++x) {
            cv::Vec3f sf = sf_copy.at<cv::Vec3f>(y, x);
            float mag = std::sqrt(sf[0]*sf[0] + sf[1]*sf[1]);
            max_flow = std::max(max_flow, mag);
        }
    }
    
    float n = 8.0f;
    
    // 对所有像素着色
    for (int y = 0; y < sf_copy.rows; ++y) {
        for (int x = 0; x < sf_copy.cols; ++x) {
            cv::Vec3f sf = sf_copy.at<cv::Vec3f>(y, x);
            
            float mag = std::sqrt(sf[0]*sf[0] + sf[1]*sf[1]);
            float dir = std::atan2(sf[1], sf[0]);
            
            float h = fmod(dir/(2.0*M_PI)+1.0, 1.0);
            float s = std::min(std::max(mag*n/max_flow, 0.0f), 1.0f);
            float v = std::min(std::max(n-s, 0.0f), 1.0f);
            
            float r, g, b;
            hsvToRgb(h, s, v, r, g, b);
            color_map.at<cv::Vec3b>(y, x) = cv::Vec3b(b*255, g*255, r*255);
        }
    }
    
    return color_map;
}

// ===========================================
// 修改现有函数：computeSceneFlowErrorMap
// 使用KITTI官方误差颜色映射
// ===========================================
cv::Mat EvaluateSceneFlow::computeSceneFlowErrorMap(const cv::Mat& pred_sf, const cv::Mat& gt_sf) {
    cv::Mat error_map(pred_sf.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    
    for (int y = 0; y < pred_sf.rows; ++y) {
        for (int x = 0; x < pred_sf.cols; ++x) {
            cv::Vec3f pred = pred_sf.at<cv::Vec3f>(y, x);
            cv::Vec3f gt = gt_sf.at<cv::Vec3f>(y, x);
            
            bool gt_valid = (!std::isnan(gt[0]) && !std::isnan(gt[1]) && !std::isnan(gt[2]));
            
            if (gt_valid) {
                cv::Vec3b val(0, 0, 0);
                
                // 计算3D欧氏距离误差
                float dx = pred[0] - gt[0];
                float dy = pred[1] - gt[1]; 
                float dz = pred[2] - gt[2];
                float scene_flow_err = std::sqrt(dx*dx + dy*dy + dz*dz);
                float scene_flow_mag = std::sqrt(gt[0]*gt[0] + gt[1]*gt[1] + gt[2]*gt[2]);
                
                // 归一化误差
                float n_err = std::min(scene_flow_err/3.0f, 20.0f*scene_flow_err/(scene_flow_mag + 1e-6f));
                
                // 应用KITTI颜色映射
                for (int i = 0; i < 10; i++) {
                    if (n_err >= LC[i][0] && n_err < LC[i][1]) {
                        val[2] = (uint8_t)LC[i][2]; // R
                        val[1] = (uint8_t)LC[i][3]; // G
                        val[0] = (uint8_t)LC[i][4]; // B
                        break;
                    }
                }
                
                // 单像素填充，不使用3x3区域
                error_map.at<cv::Vec3b>(y, x) = val;
            }
        }
    }
    return error_map;
}

void EvaluateSceneFlow::exportSceneFlowComparisonCSV(
    const std::string& csv_path,
    const std::map<std::string, std::vector<SceneFlowMetrics>>& method_results)
{
    std::ofstream file(csv_path, std::ios::trunc);
    if (!file.is_open()) return;
    
    file << "Method,EPE3d(m),AccS(%),AccR(%),Outlier(%),Runtime(ms)\n";
    
    // 遍历所有方法的结果
    for (const auto& method_pair : method_results) {
        const std::string& method_name = method_pair.first;
        const std::vector<SceneFlowMetrics>& results = method_pair.second;
        
        if (!results.empty()) {
            // 计算平均值
            double avg_EPE3d = 0, avg_AccS = 0, avg_AccR = 0, avg_Outlier = 0, avg_time = 0;
            for (const auto& metrics : results) {
                avg_EPE3d += metrics.EPE3d;
                avg_AccS += metrics.AccS;
                avg_AccR += metrics.AccR;
                avg_Outlier += metrics.Outlier;
                avg_time += metrics.time_ms;
            }
            size_t count = results.size();
            
            // 写入该方法的平均值
            file << method_name << "+Disp," << avg_EPE3d/count << "," << avg_AccS/count << "," 
                 << avg_AccR/count << "," << avg_Outlier/count << "," << avg_time/count << "\n";
        }
    }
    file.close();
}

/**
 * @brief 统一的场景流评估入口函数 - 替换原有的runEvaluation和runEvaluationBatch
 * @param method 光流方法名称
 * @param display_images 是否显示可视化
 * @param image_indices 图像索引数组（单帧传{i}，批量传{i1,i2,...}）
 * @return 评估结果数组
 */
std::vector<SceneFlowMetrics> EvaluateSceneFlow::runEvaluation(
    const std::string &method,
    bool display_images,
    const std::vector<int> &image_indices)
{
    std::vector<SceneFlowMetrics> results;
    
    // 判断是否支持真批量处理
    bool is_batch_capable = (method == "degraf_flow_interponet");
    
    // =====================================================
    // 步骤1: 数据准备
    // =====================================================
    struct ImagePairData {
        cv::Mat i1, i2;
        cv::Mat gray1, gray2;
        std::string i1_path, i2_path;
        std::string disp0_path, disp1_path;
        std::string flow_gt_path;
        std::string calib_path;
        std::string num_str;
        float fx, fy, cx, cy, baseline;
        int image_no;
    };
    
    std::vector<ImagePairData> batch_data;
    batch_data.reserve(image_indices.size());
    
    // 加载所有数据
    for (int image_no : image_indices) {
        ImagePairData data;
        data.image_no = image_no;
        
        // 路径构建
        char num[7];
        sprintf(num, "%06d", image_no);
        data.num_str = std::string(num);
        
        // 根据实际需求选择training或testing目录
        std::string base_dir = "../data/data_scene_flow/training/";  // 或者使用testing
        data.i1_path = base_dir + "image_2/" + data.num_str + "_10.png";
        data.i2_path = base_dir + "image_2/" + data.num_str + "_11.png";
        data.disp0_path = base_dir + "disp_noc_0/" + data.num_str + "_10.png";
        data.disp1_path = base_dir + "disp_noc_1/" + data.num_str + "_10.png";
        data.flow_gt_path = base_dir + "flow_noc/" + data.num_str + "_10.png";
        data.calib_path = "../data/data_scene_flow_calib/training/calib_cam_to_cam/" + data.num_str + ".txt";
        
        // 加载图像
        data.i1 = cv::imread(data.i1_path, 1);
        data.i2 = cv::imread(data.i2_path, 1);
        cv::Mat disp0 = cv::imread(data.disp0_path, cv::IMREAD_UNCHANGED);
        cv::Mat disp1 = cv::imread(data.disp1_path, cv::IMREAD_UNCHANGED);
        cv::Mat flow_gt = cv::imread(data.flow_gt_path, cv::IMREAD_UNCHANGED);
        
        // 验证数据
        if (data.i1.empty() || data.i2.empty() || disp0.empty() || disp1.empty() || flow_gt.empty()) {
            printf("❌ Input missing %06d\n", image_no);
            continue;
        }
        if (data.i1.size() != data.i2.size() || data.i1.size() != disp0.size() || 
            disp0.size() != disp1.size() || data.i1.size() != flow_gt.size()) {
            printf("❌ Size mismatch %06d\n", image_no);
            continue;
        }
        
        // 预处理灰度图
        if (data.i1.channels() == 3)
            cv::cvtColor(data.i1, data.gray1, cv::COLOR_BGR2GRAY);
        else
            data.gray1 = data.i1.clone();
        
        if (data.i2.channels() == 3)
            cv::cvtColor(data.i2, data.gray2, cv::COLOR_BGR2GRAY);
        else
            data.gray2 = data.i2.clone();
        
        // 加载标定参数
        if (!loadCameraIntrinsics(data.calib_path, data.fx, data.fy, data.cx, data.cy, data.baseline)) {
            data.fx = 721.5377f;
            data.fy = 721.5377f;
            data.cx = 609.5593f;
            data.cy = 172.8540f;
            data.baseline = 0.5371f;
            printf("Using default KITTI camera parameters for %06d\n", image_no);
        }
        
        batch_data.push_back(data);
    }
    
    if (batch_data.empty()) {
        return results;
    }
    
    // =====================================================
    // 步骤2: 光流计算（根据方法选择批量或逐帧）
    // =====================================================
    std::vector<cv::Mat> batch_flows;
    std::vector<double> individual_times;
    
    if (is_batch_capable && batch_data.size() > 1) {
        // InterpoNet批量处理
        std::vector<cv::Mat> batch_i1, batch_i2;
        std::vector<std::string> batch_num_strs;
        
        for (const auto& data : batch_data) {
            batch_i1.push_back(data.i1);
            batch_i2.push_back(data.i2);
            batch_num_strs.push_back(data.num_str);
        }
        
        double batch_start = cv::getTickCount();
        FeatureMatcher matcher;
        
        // 调用批量版本，获取特征点用于可视化（如果需要）
        std::vector<std::vector<cv::Point2f>> batch_points, batch_dst_points;
        batch_flows = matcher.degraf_flow_InterpoNet(
            batch_i1, batch_i2, batch_num_strs,
            display_images ? &batch_points : nullptr,      // 只在需要可视化时获取
            display_images ? &batch_dst_points : nullptr
        );
        
        double total_time_ms = (cv::getTickCount() - batch_start) / cv::getTickFrequency() * 1000.0;
        
        // 批量处理时，每帧时间是平均时间
        for (size_t i = 0; i < batch_flows.size(); ++i) {
            individual_times.push_back(total_time_ms / batch_flows.size());
        }
    } else {
        // 逐帧处理（可并行）
        batch_flows.resize(batch_data.size());
        individual_times.resize(batch_data.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < batch_data.size(); ++i) {
            const auto& data = batch_data[i];
            cv::Mat flow;
            
            double single_start = cv::getTickCount();
            
            // 光流估计
            if (method == "farneback")
                cv::optflow::createOptFlow_Farneback()->calc(data.gray1, data.gray2, flow);
            else if (method == "tvl1")
                cv::optflow::createOptFlow_DualTVL1()->calc(data.gray1, data.gray2, flow);
            else if (method == "deepflow")
                cv::optflow::createOptFlow_DeepFlow()->calc(data.gray1, data.gray2, flow);
            else if (method == "DISflow_fast")
                cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST)->calc(data.gray1, data.gray2, flow);
            else if (method == "DISflow_medium")
                cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM)->calc(data.gray1, data.gray2, flow);
            else if (method == "degraf_flow_rlof") {
                FeatureMatcher matcher;
                matcher.degraf_flow_RLOF(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            }
            else if (method == "degraf_flow_lk") {
                FeatureMatcher matcher;
                matcher.degraf_flow_LK(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            }
            // else if (method == "degraf_flow_interponet") {
            //     // 单帧InterpoNet重定向到RLOF（更高效）
            //     printf("Note: Single frame InterpoNet redirected to RLOF for better efficiency\n");
            //     FeatureMatcher matcher;
            //     matcher.degraf_flow_RLOF(data.i1, data.i2, flow, 127, 0.05f, true, 500.0f, 1.5f, data.num_str);
            // }
            else {
                printf("❌ Unknown optical flow method: %s\n", method.c_str());
            }
            
            double single_time = (cv::getTickCount() - single_start) / cv::getTickFrequency() * 1000.0;
            
            #pragma omp critical
            {
                batch_flows[i] = flow;
                individual_times[i] = single_time;
            }
        }
    }
    
    // =====================================================
    // 步骤3: 场景流重建和评估
    // =====================================================
    std::string csv_path = (batch_data.size() > 1) ? 
        "../data/outputs/batch_scene_flow_results.csv" : 
        "../data/outputs/scene_flow_results.csv";
    
    // 首帧写入表头
    if (batch_data[0].image_no == 0 || (batch_data.size() > 1 && batch_data[0].image_no == image_indices[0])) {
        std::ofstream header_file(csv_path, std::ios::trunc);
        header_file << "image_no,method,EPE3d,AccS(%),AccR(%),Outlier(%),valid_count,time_ms\n";
        header_file.close();
    }
    
    for (size_t i = 0; i < batch_data.size(); ++i) {
        const auto& data = batch_data[i];
        const cv::Mat& flow = batch_flows[i];
        
        if (flow.empty()) {
            printf("❌ Optical flow calculation failed %06d\n", data.image_no);
            continue;
        }
        
        // 场景流重建
        cv::Mat disp0_f32 = readKITTIDisparity(data.disp0_path);
        cv::Mat disp1_f32 = readKITTIDisparity(data.disp1_path);
        
        SceneFlowReconstructor reconstructor(data.fx, data.fy, data.cx, data.cy, data.baseline);
        cv::Mat scene_flow = reconstructor.computeSceneFlow(flow, disp0_f32, disp1_f32);
        
        if (scene_flow.empty()) {
            printf("Scene flow calculation failed %06d\n", data.image_no);
            continue;
        }
        
        // 读取GT
        cv::Mat flow_gt_processed = readKITTIFlowGT(data.flow_gt_path);
        cv::Mat gt_scene_flow = reconstructor.computeSceneFlow(flow_gt_processed, disp0_f32, disp1_f32);
        
        if (gt_scene_flow.empty() || gt_scene_flow.type() != CV_32FC3) {
            printf("GT Scene Flow reading failed %06d\n", data.image_no);
            continue;
        }
        
        // 评估
        SceneFlowMetrics metrics = evaluateSingleFrame(scene_flow, gt_scene_flow, true);
        metrics.time_ms = individual_times[i];
        
        // 新增可视化调用
        if (display_images) {
            generateSceneFlow4PanelVisualization(
                method, 
                data.image_no,
                data.i1,
                scene_flow,
                gt_scene_flow,
                "../data/outputs/kitti_results/" + method + "_scene_flow_vis_" + data.num_str + ".png"  // 添加方法名
            );
        }
        

        // 写入CSV
        writeMetricsToCSV(metrics, method, data.image_no, csv_path);
        
        // 存储结果
        results.push_back(metrics);
        all_results_.push_back(metrics);
        
        printf("✅ Frame %06d evaluated successfully\n", data.image_no);
    }
    
    // =====================================================
    // 步骤4: 批量处理时添加平均值
    // =====================================================
    if (batch_data.size() > 1 && !results.empty()) {
        double avg_EPE3d = 0, avg_AccS = 0, avg_AccR = 0, avg_Outlier = 0, avg_time = 0;
        int total_valid = 0;
        
        for (const auto& metrics : results) {
            avg_EPE3d += metrics.EPE3d;
            avg_AccS += metrics.AccS;
            avg_AccR += metrics.AccR;
            avg_Outlier += metrics.Outlier;
            avg_time += metrics.time_ms;
            total_valid += metrics.valid_count;
        }
        
        size_t count = results.size();
        avg_EPE3d /= count;
        avg_AccS /= count;
        avg_AccR /= count;
        avg_Outlier /= count;
        avg_time /= count;
        int avg_valid = total_valid / count;
        
        std::ofstream file(csv_path, std::ios::app);
        if (file.is_open()) {
            file << "AVERAGE," << method << ","
                 << avg_EPE3d << "," << avg_AccS << "," << avg_AccR << ","
                 << avg_Outlier << "," << avg_valid << "," << avg_time << "\n";
            file.close();
        }
    }

    return results;
}

/**
 * @brief 便利重载 - 保持向后兼容的单帧接口
 */
SceneFlowMetrics EvaluateSceneFlow::runEvaluation(
    const std::string &method,
    bool display_images,
    int image_no)
{
    std::vector<int> indices = {image_no};
    std::vector<SceneFlowMetrics> results = runEvaluation(method, display_images, indices);
    
    if (!results.empty()) {
        return results[0];
    } else {
        return SceneFlowMetrics();  // 返回默认构造的空结果
    }
}

// 新增：视差稠密化函数
cv::Mat EvaluateSceneFlow::densifyDisparity(const cv::Mat& sparse_disp) {
    cv::Mat dense_disp = sparse_disp.clone();
    
    // 水平方向插值
    for (int y = 0; y < dense_disp.rows; ++y) {
        std::vector<int> valid_x;
        for (int x = 0; x < dense_disp.cols; ++x) {
            float disp = dense_disp.at<float>(y, x);
            if (!std::isnan(disp) && disp > 0) {
                valid_x.push_back(x);
            }
        }
        
        // 在有效点之间插值
        for (size_t i = 0; i < valid_x.size() - 1; ++i) {
            int x1 = valid_x[i], x2 = valid_x[i + 1];
            float disp1 = dense_disp.at<float>(y, x1);
            float disp2 = dense_disp.at<float>(y, x2);
            
            for (int x = x1 + 1; x < x2; ++x) {
                float ratio = float(x - x1) / (x2 - x1);
                dense_disp.at<float>(y, x) = disp1 * (1 - ratio) + disp2 * ratio;
            }
        }
    }
    return dense_disp;
}

// 新增：场景流稠密化函数
cv::Mat EvaluateSceneFlow::densifySceneFlow(const cv::Mat& sparse_sf) {
    cv::Mat dense_sf = sparse_sf.clone();
    
    for (int y = 0; y < dense_sf.rows; ++y) {
        std::vector<int> valid_x;
        for (int x = 0; x < dense_sf.cols; ++x) {
            cv::Vec3f sf = dense_sf.at<cv::Vec3f>(y, x);
            if (!std::isnan(sf[0]) && (sf[0] != 0 || sf[1] != 0 || sf[2] != 0)) {
                valid_x.push_back(x);
            }
        }
        
        for (size_t i = 0; i < valid_x.size() - 1; ++i) {
            int x1 = valid_x[i], x2 = valid_x[i + 1];
            cv::Vec3f sf1 = dense_sf.at<cv::Vec3f>(y, x1);
            cv::Vec3f sf2 = dense_sf.at<cv::Vec3f>(y, x2);
            
            for (int x = x1 + 1; x < x2; ++x) {
                float ratio = float(x - x1) / (x2 - x1);
                dense_sf.at<cv::Vec3f>(y, x) = sf1 * (1 - ratio) + sf2 * ratio;
            }
        }
    }
    return dense_sf;
}

// 新增：KITTI标准HSV转RGB
void EvaluateSceneFlow::hsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    float c = v * s;
    float h2 = 6.0f * h;
    float x = c * (1.0f - fabsf(fmodf(h2, 2.0f) - 1.0f));

    if (0<=h2 && h2<1)       { r = c; g = x; b = 0; }
    else if (1<=h2 && h2<2)  { r = x; g = c; b = 0; }
    else if (2<=h2 && h2<3)  { r = 0; g = c; b = x; }
    else if (3<=h2 && h2<4)  { r = 0; g = x; b = c; }
    else if (4<=h2 && h2<5)  { r = x; g = 0; b = c; }
    else if (5<=h2 && h2<=6) { r = c; g = 0; b = x; }
    else                     { r = 0; g = 0; b = 0; }
}