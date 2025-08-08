#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

// 从 KITTI PNG 文件读取 GT 光流，转换为 CV_32FC2 格式
Mat readKittiGroundTruth(const string& png_path) {
    Mat image = imread(png_path, IMREAD_UNCHANGED);
    if (image.empty()) {
        cerr << "[ERROR] Failed to read image: " << png_path << endl;
        return {};
    }

    Mat gt(image.rows, image.cols, CV_32FC2);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            Vec3w val = image.at<Vec3w>(y, x);  // 3 channel uint16

            Vec2f flow;
            if (val[0] > 0) {
                flow[0] = ((float)val[2] - 32768.0f) / 64.0f;
                flow[1] = ((float)val[1] - 32768.0f) / 64.0f;
            } else {
                flow[0] = std::numeric_limits<float>::quiet_NaN();
                flow[1] = std::numeric_limits<float>::quiet_NaN();
            }
            gt.at<Vec2f>(y, x) = flow;
        }
    }
    return gt;
}

// 保存 .flo 文件
void saveFlowToFlo(const string& filename, const Mat& flow) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "[ERROR] Failed to open output file: " << filename << endl;
        return;
    }

    // 写入头部
    file.write("PIEH", 4);

    int32_t width = flow.cols;
    int32_t height = flow.rows;
    file.write(reinterpret_cast<char*>(&width), sizeof(int32_t));
    file.write(reinterpret_cast<char*>(&height), sizeof(int32_t));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec2f f = flow.at<Vec2f>(y, x);
            file.write(reinterpret_cast<char*>(&f[0]), sizeof(float));
            file.write(reinterpret_cast<char*>(&f[1]), sizeof(float));
        }
    }

    file.close();
}

// 使用 OpenCV 的 glob 和 imread 来批量处理 PNG → FLO
void batchConvert(const string& input_dir, const string& output_dir) {
    vector<String> png_files;
    glob(input_dir + "/*.png", png_files, false);

    if (png_files.empty()) {
        cerr << "[ERROR] No PNG files found in: " << input_dir << endl;
        return;
    }

    // 创建输出文件夹（用 system 命令最通用）
    string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    for (const auto& png_path : png_files) {
        size_t name_start = png_path.find_last_of("/\\") + 1;
        string name = png_path.substr(name_start);
        string output_name = name.substr(0, name.size() - 4) + ".flo";

        Mat flow = readKittiGroundTruth(png_path);
        if (!flow.empty()) {
            saveFlowToFlo(output_dir + "/" + output_name, flow);
            cout << "[✓] Converted: " << name << " → " << output_name << endl;
        }
    }
}

// 主函数
int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_png_dir> <output_flo_dir>" << endl;
        return -1;
    }

    string input_dir = argv[1];
    string output_dir = argv[2];

    batchConvert(input_dir, output_dir);
    return 0;
}
