/*!
\file utils.h
\brief 统一TCP服务器通信工具函数头文件
\author Gang Wang, Durham University 2025
*/

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <vector>
#include <string>

namespace UnifiedTCPUtils {

/**
 * @brief 调用统一TCP服务器进行文件路径传输的pipeline处理
 * @param batch_img1_paths 第一帧图像路径批量
 * @param batch_img2_paths 第二帧图像路径批量  
 * @param batch_points_paths 特征点文件路径批量
 * @param batch_edges_paths 边缘文件路径批量
 * @param batch_output_paths 输出路径批量
 * @return 成功返回true，失败返回false
 */
bool callUnifiedTCP_file(
    const std::vector<std::string>& batch_img1_paths,
    const std::vector<std::string>& batch_img2_paths,
    const std::vector<std::string>& batch_points_paths,
    const std::vector<std::string>& batch_edges_paths,
    const std::vector<std::string>& batch_output_paths);

/**
 * @brief 从RAFT生成的matches文件解析特征点对
 * @param matches_path matches文件路径
 * @param src_points 输出源图像特征点
 * @param dst_points 输出目标图像特征点
 */
void parseMatchesFile(const std::string& matches_path,
                     std::vector<cv::Point2f>& src_points,
                     std::vector<cv::Point2f>& dst_points);

/**
 * @brief 保存边缘检测结果为二进制.dat文件
 * @param edge_map 边缘检测Mat
 * @param filename 输出文件名
 */
void save_edge_dat(const cv::Mat& edge_map, const std::string& filename);

/**
 * @brief 读取.flo光流文件
 * @param path .flo文件路径
 * @return 光流Mat (CV_32FC2格式)
 */
cv::Mat readOpticalFlowFile(const std::string& path);

/**
 * @brief 保存特征点为RAFT格式的txt文件
 * @param points 特征点向量
 * @param filepath 输出文件路径
 */
void savePointsToFile(const std::vector<cv::Point2f>& points, const std::string& filepath);

/**
 * @brief 检查文件是否存在且时间戳比源文件新
 * @param targetFile 目标文件路径
 * @param sourceFile 源文件路径
 * @return 是否为最新文件
 */
bool isFileUpToDate(const std::string& targetFile, const std::string& sourceFile);

/**
 * @brief 检查特征点缓存是否有效
 * @param pointsFile 特征点文件路径
 * @param imageFile 图像文件路径
 * @return 缓存是否有效
 */
bool isPointsCacheValid(const std::string& pointsFile, const std::string& imageFile);

/**
 * @brief 检查边缘检测缓存是否有效
 * @param edgeFile 边缘文件路径
 * @param imageFile 图像文件路径
 * @return 缓存是否有效
 */
bool isEdgeCacheValid(const std::string& edgeFile, const std::string& imageFile);

/**
 * @brief 加载已缓存的特征点
 * @param pointsFile 特征点文件路径
 * @return 特征点向量
 */
std::vector<cv::Point2f> loadCachedPoints(const std::string& pointsFile);

} // namespace UnifiedTCPUtils

#endif // UTILS_H