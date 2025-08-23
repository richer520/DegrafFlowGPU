#!/usr/bin/env python3
"""
DeGraF RAFT Matcher - Optimized Batch Processing
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
import time
from PIL import Image
import gc
import subprocess
import torch.nn.functional as F
from collections import defaultdict
from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 全局缓存
_image_cache = {}
def load_image(imfile):
    """加载图像并转换为张量 - 带缓存"""
    if imfile in _image_cache:
        return _image_cache[imfile].clone()
    
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    img_tensor = img_tensor[None].to(DEVICE)
    
    _image_cache[imfile] = img_tensor
    return img_tensor

def load_points(points_file):
    """加载特征点文件"""
    points = []
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                points.append([x, y])
    return np.array(points, dtype=np.float32)

def sample_flow_at_points(flow, points, img_width, img_height):
    """在特征点位置采样光流值 - GPU加速版本"""
    points_tensor = torch.tensor(points, dtype=torch.float32, device=flow.device)
    flow_height, flow_width = flow.shape[2], flow.shape[3]
    
    scale_x = flow_width / img_width
    scale_y = flow_height / img_height
    
    fx = points_tensor[:, 0] * scale_x
    fy = points_tensor[:, 1] * scale_y
    fx_norm = (fx / (flow_width - 1)) * 2 - 1
    fy_norm = (fy / (flow_height - 1)) * 2 - 1
    
    grid = torch.stack([fx_norm, fy_norm], dim=-1).unsqueeze(0).unsqueeze(2)
    sampled_flow = F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
    sampled_flow = sampled_flow.squeeze()
    
    if sampled_flow.dim() == 1:
        sampled_flow = sampled_flow.unsqueeze(1)
    
    flow_x = sampled_flow[0, :]
    flow_y = sampled_flow[1, :]
    
    x2 = points_tensor[:, 0] + flow_x / scale_x
    y2 = points_tensor[:, 1] + flow_y / scale_y
    
    matches_tensor = torch.stack([
        points_tensor[:, 0], points_tensor[:, 1], x2, y2
    ], dim=1)
    
    return matches_tensor.cpu().numpy().astype(np.float32)

def sample_flow_at_points_batch_optimized(flow_batch, points_list, img_width, img_height):
    """优化的批量采样 - 按特征点数量分组处理"""
    batch_size = flow_batch.shape[0]
    flow_height, flow_width = flow_batch.shape[2], flow_batch.shape[3]
    
    scale_x = flow_width / img_width
    scale_y = flow_height / img_height
    
    # 按特征点数量分组
    groups = defaultdict(list)
    for i, points in enumerate(points_list):
        point_count = len(points)
        groups[point_count].append((i, points))
    
    # 初始化结果列表
    matches_list = [None] * batch_size
    
    # 分组处理
    for point_count, group_items in groups.items():
        if point_count == 0:
            # 空特征点直接返回空结果
            for idx, _ in group_items:
                matches_list[idx] = np.array([]).reshape(0, 4)
            continue
        
        if len(group_items) == 1:
            # 单个图像，逐个处理
            idx, points = group_items[0]
            single_flow = flow_batch[idx:idx+1]
            single_matches = sample_flow_at_points(single_flow, points.cpu().numpy(), img_width, img_height)
            matches_list[idx] = single_matches
        
        else:
            # 多个图像，批量处理
            indices = [idx for idx, _ in group_items]
            group_points = [points for _, points in group_items]
            group_flows = flow_batch[indices]
            
            # 批量处理相同数量的特征点
            all_points = torch.stack(group_points)
            
            fx = all_points[:, :, 0] * scale_x
            fy = all_points[:, :, 1] * scale_y
            fx_norm = (fx / (flow_width - 1)) * 2 - 1
            fy_norm = (fy / (flow_height - 1)) * 2 - 1
            
            grid = torch.stack([fx_norm, fy_norm], dim=-1).unsqueeze(2)
            sampled_flow = F.grid_sample(group_flows, grid, mode='bilinear', align_corners=True)
            sampled_flow = sampled_flow.squeeze(3)
            
            flow_x = sampled_flow[:, 0, :]
            flow_y = sampled_flow[:, 1, :]
            
            x2 = all_points[:, :, 0] + flow_x / scale_x
            y2 = all_points[:, :, 1] + flow_y / scale_y
            
            matches_batch = torch.stack([
                all_points[:, :, 0], all_points[:, :, 1], x2, y2
            ], dim=2)
            
            # 将结果放回对应位置
            group_matches = matches_batch.cpu().numpy().astype(np.float32)
            for i, idx in enumerate(indices):
                matches_list[idx] = group_matches[i]
    
    return matches_list

def save_matches(matches, output_file):
    """保存匹配文件"""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        for match in matches:
            f.write(f"{match[0]:.6f} {match[1]:.6f} {match[2]:.6f} {match[3]:.6f}\n")

class RAFTModelManager:
    """GPU内存优化的模型管理器"""
    _instance = None
    _model_loaded = False
    _model = None
    _model_path = None
    _model_config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAFTModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path, small=False, mixed_precision=False, alternate_corr=False):
        """智能模型加载和复用"""
        current_config = (model_path, small, mixed_precision, alternate_corr)
        
        if self._model_loaded and self._model_path == model_path and self._model_config == current_config:
            return self._model
        
        self.cleanup()
        
        args = argparse.Namespace()
        args.model = model_path
        args.small = small
        args.mixed_precision = mixed_precision
        args.alternate_corr = alternate_corr
        
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(model_path))
        model = model.module
        model.to(DEVICE)
        model.eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._model = model
        self._model_path = model_path
        self._model_config = current_config
        self._model_loaded = True
        
        return self._model
    
    def cleanup(self):
        """清理模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            self._model_path = None
            self._model_config = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()

# 全局模型管理器实例
_model_manager = RAFTModelManager()

def load_model(model_path, small=False, mixed_precision=False, alternate_corr=False):
    """加载并缓存RAFT模型"""
    global _model_manager
    return _model_manager.load_model(model_path, small, mixed_precision, alternate_corr)

def run_raft_matching_batch(batch_data, model_path='/app/models/raft-kitti.pth',
                          small=False, mixed_precision=False, alternate_corr=False):
    """
    批量执行RAFT光流匹配 - GPU性能优化版本
    """
    model = _model_manager.load_model(model_path, small, mixed_precision, alternate_corr)
    
    # 优化的分组和预加载
    size_groups = {}
    for i, item in enumerate(batch_data):
        image1 = load_image(item['image1_path'])
        image2 = load_image(item['image2_path'])
        points = load_points(item['points_path'])
        points_gpu = torch.tensor(points, dtype=torch.float32, device=DEVICE)
        
        img_height, img_width = image1.shape[2], image1.shape[3]
        size_key = (img_width, img_height)
        
        if size_key not in size_groups:
            size_groups[size_key] = []
        
        size_groups[size_key].append({
            'index': i,
            'image1': image1,
            'image2': image2,
            'points': points,
            'points_gpu': points_gpu,  
            'data': item
        })

    with torch.no_grad():
        all_matches = [None] * len(batch_data)
        
        for size_key, group in size_groups.items():
            group_size = len(group)
            
            # 动态批量大小策略
            if group_size <= 4:
                batch_size = group_size
            elif group_size <= 12:
                batch_size = min(8, group_size)
            elif group_size <= 32:
                batch_size = min(12, group_size)
            else:
                batch_size = min(16, group_size)
            
            for start_idx in range(0, group_size, batch_size):
                end_idx = min(start_idx + batch_size, group_size)
                current_batch = group[start_idx:end_idx]
                
                # 预分配内存
                batch_size_actual = len(current_batch)
                sample_tensor = current_batch[0]['image1'].cpu()
                img_shape = sample_tensor.shape
                
                batch_tensor1 = torch.empty((batch_size_actual, img_shape[1], img_shape[2], img_shape[3]), 
                                        dtype=torch.float32, device=DEVICE)
                batch_tensor2 = torch.empty((batch_size_actual, img_shape[1], img_shape[2], img_shape[3]), 
                                        dtype=torch.float32, device=DEVICE)

                for i, item in enumerate(current_batch):
                    batch_tensor1[i] = item['image1'].squeeze(0)
                    batch_tensor2[i] = item['image2'].squeeze(0)

                torch.cuda.synchronize()

                # Padding处理
                padder = InputPadder(batch_tensor1.shape)
                batch_tensor1_pad, batch_tensor2_pad = padder.pad(batch_tensor1, batch_tensor2)
                
                try:
                    # RAFT推理
                    flow_low, flow_up = model(batch_tensor1_pad, batch_tensor2_pad, iters=12, test_mode=True)
                    
                    # 优化的批量采样
                    points_list = [item['points_gpu'] for item in current_batch]
                    matches_list = sample_flow_at_points_batch_optimized(flow_up, points_list, size_key[0], size_key[1])

                    # 保存结果
                    for i, item in enumerate(current_batch):
                        save_matches(matches_list[i], item['data']['output_path'])
                        all_matches[item['index']] = len(matches_list[i])

                    # 清理tensor
                    del batch_tensor1, batch_tensor2, flow_low, flow_up

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # 降级处理：分成更小的批次
                        for item in current_batch:
                            img1_pad, img2_pad = padder.pad(item['image1'], item['image2'])
                            flow_low, flow_up = model(img1_pad, img2_pad, iters=20, test_mode=True)
                            
                            matches = sample_flow_at_points(flow_up, item['points'], size_key[0], size_key[1])
                            save_matches(matches, item['data']['output_path'])
                            all_matches[item['index']] = len(matches)
                            
                            del flow_low, flow_up
                    else:
                        raise e
                
                torch.cuda.empty_cache()

    return len(batch_data)

def force_cleanup():
    """强制清理所有资源的外部接口"""
    global _model_manager
    _model_manager.cleanup()

def main():
    parser = argparse.ArgumentParser(description='DeGraF RAFT Matcher - Batch Processing Only')
    parser.add_argument('--model', required=True, help="RAFT model checkpoint path")
    parser.add_argument('--batch_config', required=True, help="Batch configuration file")
    
    args = parser.parse_args()
    
    # 这里可以添加批量配置文件读取逻辑
    print("DeGraF RAFT Matcher - Optimized for batch processing")
    print("Use the batch TCP server or call run_raft_matching_batch() directly")

if __name__ == '__main__':
    main()