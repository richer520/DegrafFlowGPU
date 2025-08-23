/*!
\file utils.cpp
\brief 统一TCP服务器通信工具函数
\author Gang Wang, Durham University 2025
*/

#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <json/json.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace UnifiedTCPUtils {

bool callUnifiedTCP_file(
    const std::vector<std::string>& batch_img1_paths,
    const std::vector<std::string>& batch_img2_paths,
    const std::vector<std::string>& batch_points_paths,
    const std::vector<std::string>& batch_edges_paths,
    const std::vector<std::string>& batch_output_paths) {
    
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return false;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(9999);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        return false;
    }
    
    Json::Value request;
    request["type"] = "unified_file_pipeline";
    request["batch_size"] = (int)batch_img1_paths.size();
    
    Json::Value img1_paths_array(Json::arrayValue);
    Json::Value img2_paths_array(Json::arrayValue);
    Json::Value points_paths_array(Json::arrayValue);
    Json::Value edges_paths_array(Json::arrayValue);
    Json::Value output_paths_array(Json::arrayValue);
    
    for (size_t i = 0; i < batch_img1_paths.size(); ++i) {
        img1_paths_array.append(batch_img1_paths[i]);
        img2_paths_array.append(batch_img2_paths[i]);
        points_paths_array.append(batch_points_paths[i]);
        edges_paths_array.append(batch_edges_paths[i]);
        output_paths_array.append(batch_output_paths[i]);
    }
    
    request["image1_paths"] = img1_paths_array;
    request["image2_paths"] = img2_paths_array;
    request["points_paths"] = points_paths_array;
    request["edges_paths"] = edges_paths_array;
    request["output_paths"] = output_paths_array;
    
    Json::FastWriter writer;
    std::string request_str = writer.write(request);
    if (request_str.back() != '\n') {
        request_str += '\n';
    }
    
    send(sock, request_str.c_str(), request_str.length(), 0);
    
    std::string response_str;
    char buffer[8192];
    int bytes_received;
    
    while ((bytes_received = recv(sock, buffer, sizeof(buffer), 0)) > 0) {
        response_str.append(buffer, bytes_received);
        if (!response_str.empty() && response_str.back() == '\n') break;
    }
    
    close(sock);
    
    if (bytes_received <= 0) {
        return false;
    }
    
    Json::Value response;
    Json::CharReaderBuilder reader_builder;
    std::istringstream response_stream(response_str);
    std::string parse_errors;
    
    if (!Json::parseFromStream(reader_builder, response_stream, &response, &parse_errors)) {
        return false;
    }
    
    return response["status"].asString() == "success";
}

void parseMatchesFile(const std::string& matches_path,
                     std::vector<cv::Point2f>& src_points,
                     std::vector<cv::Point2f>& dst_points) {
    src_points.clear();
    dst_points.clear();

    std::ifstream file(matches_path);
    if (!file.is_open()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x0, y0, x1, y1;
        if (iss >> x0 >> y0 >> x1 >> y1) {
            src_points.emplace_back(x0, y0);
            dst_points.emplace_back(x1, y1);
        }
    }
    file.close();
}

void save_edge_dat(const cv::Mat& edge_map, const std::string& filename) {
    cv::Mat edge_float;
    if (edge_map.type() != CV_32F)
        edge_map.convertTo(edge_float, CV_32F);
    else
        edge_float = edge_map;

    if (edge_float.channels() > 1)
        cv::cvtColor(edge_float, edge_float, cv::COLOR_BGR2GRAY);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(edge_float.data),
              edge_float.rows * edge_float.cols * sizeof(float));
    ofs.close();
}

cv::Mat readOpticalFlowFile(const std::string& path) {
    std::ifstream file(path.c_str(), std::ios_base::binary);
    if (!file.good()) {
        return cv::Mat();
    }

    float magic;
    file.read((char *)&magic, sizeof(float));
    if (magic != 202021.25f) {
        return cv::Mat();
    }

    int width, height;
    file.read((char *)&width, sizeof(int));
    file.read((char *)&height, sizeof(int));

    cv::Mat flow(height, width, CV_32FC2);
    file.read((char *)flow.data, width * height * 2 * sizeof(float));
    file.close();

    return flow;
}

void savePointsToFile(const std::vector<cv::Point2f>& points, const std::string& filepath) {
    std::ofstream file(filepath);
    if (file.is_open()) {
        for (const auto& point : points) {
            file << point.x << " " << point.y << "\n";
        }
        file.close();
    }
}

bool isFileUpToDate(const std::string& targetFile, const std::string& sourceFile) {
    if (!cv::utils::fs::exists(targetFile)) {
        return false;
    }
    
    struct stat targetStat, sourceStat;
    if (stat(targetFile.c_str(), &targetStat) != 0 || stat(sourceFile.c_str(), &sourceStat) != 0) {
        return false;
    }
    
    return targetStat.st_mtime >= sourceStat.st_mtime;
}

bool isPointsCacheValid(const std::string& pointsFile, const std::string& imageFile) {
    return isFileUpToDate(pointsFile, imageFile);
}

bool isEdgeCacheValid(const std::string& edgeFile, const std::string& imageFile) {
    return isFileUpToDate(edgeFile, imageFile);
}

std::vector<cv::Point2f> loadCachedPoints(const std::string& pointsFile) {
    std::vector<cv::Point2f> points;
    std::ifstream file(pointsFile);
    if (!file.is_open()) {
        return points;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y;
        if (iss >> x >> y) {
            points.emplace_back(x, y);
        }
    }
    file.close();
    return points;
}

} // namespace UnifiedTCPUtils