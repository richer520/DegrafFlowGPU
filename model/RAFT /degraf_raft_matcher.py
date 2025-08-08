#!/usr/bin/env python3
"""
DeGraF RAFT Matcher
用于从RAFT稠密光流中提取稀疏特征点的匹配关系
输入：图像对 + 稀疏特征点
输出：matches.txt (格式: x1 y1 x2 y2)
"""

import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    """加载图像并转换为张量"""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_points(points_file):
    """
    加载特征点文件
    格式：每行 x y
    返回：numpy array of shape (N, 2)
    """
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                points.append([x, y])
    return np.array(points, dtype=np.float32)

def sample_flow_at_points(flow, points, img_width, img_height):
    """
    在特征点位置采样光流值
    
    Args:
        flow: 光流张量 [1, 2, H, W] (RAFT输出)
        points: 特征点坐标 numpy array (N, 2)
        img_width, img_height: 原始图像尺寸
    
    Returns:
        matches: numpy array (N, 4) 格式为 [x1, y1, x2, y2]
    """
    flow_np = flow[0].cpu().numpy()  # [2, H, W]
    flow_height, flow_width = flow_np.shape[1], flow_np.shape[2]
    
    # 计算缩放比例（处理padding后的尺寸差异）
    scale_x = flow_width / img_width
    scale_y = flow_height / img_height
    
    matches = []
    for point in points:
        x, y = point
        
        # 将点坐标缩放到flow尺寸
        fx = x * scale_x
        fy = y * scale_y
        
        # 确保坐标在有效范围内
        fx = np.clip(fx, 0, flow_width - 1)
        fy = np.clip(fy, 0, flow_height - 1)
        
        # 使用双线性插值采样光流
        ix = int(fx)
        iy = int(fy)
        
        # 计算插值权重
        dx = fx - ix
        dy = fy - iy
        
        # 边界处理
        ix1 = min(ix + 1, flow_width - 1)
        iy1 = min(iy + 1, flow_height - 1)
        
        # 双线性插值
        flow_x = (1 - dx) * (1 - dy) * flow_np[0, iy, ix] + \
                 dx * (1 - dy) * flow_np[0, iy, ix1] + \
                 (1 - dx) * dy * flow_np[0, iy1, ix] + \
                 dx * dy * flow_np[0, iy1, ix1]
                 
        flow_y = (1 - dx) * (1 - dy) * flow_np[1, iy, ix] + \
                 dx * (1 - dy) * flow_np[1, iy, ix1] + \
                 (1 - dx) * dy * flow_np[1, iy1, ix] + \
                 dx * dy * flow_np[1, iy1, ix1]
        
        # 计算目标点坐标
        x2 = x + flow_x / scale_x  # 将flow缩放回原始图像尺寸
        y2 = y + flow_y / scale_y
        
        matches.append([x, y, x2, y2])
    
    return np.array(matches, dtype=np.float32)

def save_matches(matches, output_file):
    """
    保存匹配文件
    格式：x1 y1 x2 y2 (每行)
    """
    with open(output_file, 'w') as f:
        for match in matches:
            # 格式：x1 y1 x2 y2
            f.write(f"{match[0]:.6f} {match[1]:.6f} {match[2]:.6f} {match[3]:.6f}\n")

# 全局模型变量，避免重复加载
_global_model = None
_global_model_path = None

def load_model(model_path, small=False, mixed_precision=False, alternate_corr=False):
    """加载并缓存RAFT模型"""
    global _global_model, _global_model_path
    
    if _global_model is None or _global_model_path != model_path:
        print(f"Loading RAFT model from {model_path}")
        
        # 创建参数对象，模拟argparse的命名空间
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=model_path)
        parser.add_argument('--small', action='store_true', default=small)
        parser.add_argument('--mixed_precision', action='store_true', default=mixed_precision)
        parser.add_argument('--alternate_corr', action='store_true', default=alternate_corr)
        
        # 创建args对象
        args = argparse.Namespace()
        args.model = model_path
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = alternate_corr
        
        # 初始化模型
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path))
        model = model.module
        model.to(DEVICE)
        model.eval()
        
        _global_model = model
        _global_model_path = model_path
        print("Model loaded successfully")
    
    return _global_model

def run_raft_matching(image1_path, image2_path, points_path, output_path, 
                      model_path='/app/models/raft-kitti.pth',
                      small=False, mixed_precision=False, alternate_corr=False):
    """
    执行RAFT光流匹配（供外部调用的接口）
    
    Args:
        image1_path: 第一张图像路径
        image2_path: 第二张图像路径
        points_path: 特征点文件路径
        output_path: 输出匹配文件路径
        model_path: RAFT模型路径
        small: 是否使用小模型
        mixed_precision: 是否使用混合精度
        alternate_corr: 是否使用高效相关实现
    """
    # 加载模型（会自动缓存）
    model = load_model(model_path, small, mixed_precision, alternate_corr)
    
    print(f"Processing: {image1_path} -> {image2_path}")
    print(f"Points: {points_path}")
    
    # 加载图像
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)
    
    # 获取原始图像尺寸
    img_height, img_width = image1.shape[2], image1.shape[3]
    
    # 加载特征点
    points = load_points(points_path)
    print(f"Loaded {len(points)} feature points")
    
    with torch.no_grad():
        # Padding处理
        padder = InputPadder(image1.shape)
        image1_padded, image2_padded = padder.pad(image1, image2)
        
        # RAFT推理
        print("Running RAFT inference...")
        flow_low, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
        
        # 在特征点位置采样光流
        print("Sampling flow at feature points...")
        matches = sample_flow_at_points(flow_up, points, img_width, img_height)
        
        # 保存匹配结果
        print(f"Saving matches to {output_path}")
        save_matches(matches, output_path)
        
        print(f"Successfully processed {len(matches)} matches")
        return len(matches)

def process_image_pair(args):
    """处理一对图像和对应的特征点（命令行接口）"""
    return run_raft_matching(
        args.image1, 
        args.image2, 
        args.points, 
        args.output,
        args.model,
        args.small,
        args.mixed_precision,
        args.alternate_corr
    )

def main():
    parser = argparse.ArgumentParser(description='DeGraF RAFT Matcher')
    parser.add_argument('--model', required=True, help="RAFT model checkpoint path")
    parser.add_argument('--image1', required=True, help="First image path")
    parser.add_argument('--image2', required=True, help="Second image path")
    parser.add_argument('--points', required=True, help="Feature points file path")
    parser.add_argument('--output', required=True, help="Output matches file path")
    parser.add_argument('--small', action='store_true', help='Use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation')
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.image1):
        print(f"Error: Image file not found: {args.image1}")
        sys.exit(1)
    if not os.path.exists(args.image2):
        print(f"Error: Image file not found: {args.image2}")
        sys.exit(1)
    if not os.path.exists(args.points):
        print(f"Error: Points file not found: {args.points}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理图像对
    process_image_pair(args)

if __name__ == '__main__':
    main()