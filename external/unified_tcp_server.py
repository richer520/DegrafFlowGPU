#!/usr/bin/env python3

import socket
import json
import time
import os
import sys
import numpy as np
import torch
import cv2
import gc
import tempfile
from io import BytesIO

sys.path.append('/app/external/RAFT')
sys.path.append('/app/external/RAFT/core')
sys.path.append('/app/external/InterpoNet')
sys.path.append('/app/external/InterpoNet_Pytorch')
sys.path.append('/app')

from RAFT.degraf_raft_matcher import RAFTModelManager
from InterpoNet_Pytorch.interpoNet_pytorch import InterpoNetModelManager, run_interponet_inference_batch
from RAFT.core.raft import RAFT
from InterpoNet_Pytorch.interpoNet_pytorch import run_interponet_inference
from utils.utils import InputPadder
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnifiedTCPServer:
    
    def __init__(self, port=9999):
        self.port = port
        self.request_count = 0
        self.cleanup_interval = 10
        
        self.raft_model_path = '/app/external/RAFT/models/raft-kitti.pth'
        self.interponet_model_path = '/app/external/InterpoNet_Pytorch/checkpoints_pytorch/converted_from_tf.pth'
        
        self.raft_manager = RAFTModelManager()
        self.interponet_manager = InterpoNetModelManager()

    def _load_models(self):
        if not hasattr(self, '_models_loaded'):
            self.raft_model = self.raft_manager.load_model(self.raft_model_path)
            self.interponet_model = self.interponet_manager.load_model(self.interponet_model_path, DEVICE)
            self._models_loaded = True

    def _load_images_to_gpu(self, image_paths):
        batch_tensors = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
            batch_tensors.append(img_tensor)
        return batch_tensors

    def _load_points_from_files(self, points_paths):
        batch_points = []
        for path in points_paths:
            points = []
            if os.path.exists(path):
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            x, y = map(float, line.split())
                            points.append([x, y])
            points_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)
            batch_points.append(points_tensor)
        return batch_points

    def _load_edges_from_files(self, edges_paths, img1_paths):
        batch_edges = []
        for edges_path, img1_path in zip(edges_paths, img1_paths):
            if os.path.exists(edges_path):
                img = cv2.imread(img1_path)
                if img is not None:
                    height, width = img.shape[:2]
                    
                    with open(edges_path, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=np.float32)
                        # 创建可写的numpy数组副本
                        edges_array = data.copy().reshape(height, width)
                        edges_tensor = torch.from_numpy(edges_array).to(DEVICE)
                        batch_edges.append(edges_tensor)
                else:
                    batch_edges.append(torch.zeros(375, 1242, device=DEVICE))
            else:
                batch_edges.append(torch.zeros(375, 1242, device=DEVICE))
        return batch_edges

    def _batch_raft_inference(self, batch_imgs1, batch_imgs2, batch_points, temp_dir):
        matches_list = []
        matches_paths = []
        
        with torch.no_grad():
            for i, (img1, img2, points) in enumerate(zip(batch_imgs1, batch_imgs2, batch_points)):
                if len(points) == 0:
                    matches_tensor = torch.empty((0, 4), device=DEVICE)
                    matches_list.append(matches_tensor)
                else:
                    padder = InputPadder(img1.shape)
                    img1_pad, img2_pad = padder.pad(img1, img2)
                    
                    flow_low, flow_up = self.raft_model(img1_pad, img2_pad, iters=12, test_mode=True)
                    
                    matches_tensor = self._sample_flow_at_points_gpu(flow_up, points, img1.shape[3], img1.shape[2])
                    matches_list.append(matches_tensor)
                    
                    del img1_pad, img2_pad, flow_low, flow_up
                
                # Save matches to file for InterpoNet
                matches_path = os.path.join(temp_dir, f'matches_{i}.txt')
                matches_cpu = matches_tensor.cpu().numpy()
                with open(matches_path, 'w') as f:
                    for match in matches_cpu:
                        f.write(f"{match[0]:.6f} {match[1]:.6f} {match[2]:.6f} {match[3]:.6f}\n")
                matches_paths.append(matches_path)
        
        return matches_list, matches_paths

    def _sample_flow_at_points_gpu(self, flow, points, img_width, img_height):
        if len(points) == 0:
            return torch.empty((0, 4), device=DEVICE)
        
        flow_height, flow_width = flow.shape[2], flow.shape[3]
        
        scale_x = flow_width / img_width
        scale_y = flow_height / img_height
        
        fx = points[:, 0] * scale_x
        fy = points[:, 1] * scale_y
        fx_norm = (fx / (flow_width - 1)) * 2 - 1
        fy_norm = (fy / (flow_height - 1)) * 2 - 1
        
        grid = torch.stack([fx_norm, fy_norm], dim=-1).unsqueeze(0).unsqueeze(2)
        sampled_flow = F.grid_sample(flow, grid, mode='bilinear', align_corners=True)
        sampled_flow = sampled_flow.squeeze()
        
        if sampled_flow.dim() == 1:
            sampled_flow = sampled_flow.unsqueeze(1)
        
        flow_x = sampled_flow[0, :]
        flow_y = sampled_flow[1, :]
        
        x2 = points[:, 0] + flow_x / scale_x
        y2 = points[:, 1] + flow_y / scale_y
        
        matches_tensor = torch.stack([
            points[:, 0], points[:, 1], x2, y2
        ], dim=1)
        
        return matches_tensor

    def _batch_interponet_inference(self, batch_imgs1, batch_imgs2, matches_list, batch_edges, output_paths, matches_paths):
        temp_dir = tempfile.mkdtemp()
        
        try:
            batch_data = []
            original_sizes = []
            
            for i, (img1, img2, edges) in enumerate(zip(batch_imgs1, batch_imgs2, batch_edges)):
                img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                img1_path = os.path.join(temp_dir, f'img1_{i}.png')
                img2_path = os.path.join(temp_dir, f'img2_{i}.png')
                edges_path = os.path.join(temp_dir, f'edges_{i}.dat')
                
                cv2.imwrite(img1_path, img1_np)
                cv2.imwrite(img2_path, img2_np)
                
                edges_cpu = edges.cpu().numpy()
                with open(edges_path, 'wb') as f:
                    f.write(edges_cpu.astype(np.float32).tobytes())
                
                batch_data.append({
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'edges_path': edges_path,
                    'matches_path': matches_paths[i],  # Use RAFT-generated matches file
                    'output_path': output_paths[i],
                    'ba_matches_path': None,
                    'apply_variational': True,
                    'width': img1_np.shape[1],
                    'height': img1_np.shape[0]
                })
                
                original_sizes.append((img1_np.shape[1], img1_np.shape[0]))
            
            success = run_interponet_inference_batch(
                batch_data=batch_data,
                model_filename=self.interponet_model_path,
                original_sizes=original_sizes,
                downscale=8,
                dataset_type='kitti',
                device=DEVICE
            )
            
            return success
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    # def handle_unified_file_pipeline(self, request_data):
    #     self._load_models()
        
    #     batch_size = request_data['batch_size']
    #     img1_paths = request_data['image1_paths']
    #     img2_paths = request_data['image2_paths']
    #     points_paths = request_data['points_paths']
    #     edges_paths = request_data['edges_paths']
    #     output_paths = request_data['output_paths']
        
    #     batch_imgs1 = self._load_images_to_gpu(img1_paths)
    #     batch_imgs2 = self._load_images_to_gpu(img2_paths)
    #     batch_points = self._load_points_from_files(points_paths)
    #     batch_edges = self._load_edges_from_files(edges_paths, img1_paths)
        
    #     # Create temporary directory for matches files
    #     temp_dir = tempfile.mkdtemp()
        
    #     try:
    #         matches_list, matches_paths = self._batch_raft_inference(batch_imgs1, batch_imgs2, batch_points, temp_dir)
            
    #         success = self._batch_interponet_inference(batch_imgs1, batch_imgs2, matches_list, batch_edges, output_paths, matches_paths)
            
    #         del batch_imgs1, batch_imgs2, batch_points, batch_edges, matches_list
    #         torch.cuda.empty_cache()
            
    #         return {
    #             'status': 'success' if success else 'error',
    #             'batch_size': batch_size,
    #             'message': f'Unified file pipeline completed {batch_size} images' if success else 'Pipeline processing failed'
    #         }
        
    #     finally:
    #         # Clean up matches temporary directory
    #         import shutil
    #         shutil.rmtree(temp_dir, ignore_errors=True)

    def handle_unified_file_pipeline(self, request_data):
        """Step 3: Production-ready complete pipeline with InterpoNet batch inference"""
        
        batch_size = request_data['batch_size']
        img1_paths = request_data['image1_paths']
        img2_paths = request_data['image2_paths']
        points_paths = request_data['points_paths']
        edges_paths = request_data['edges_paths']
        output_paths = request_data['output_paths']
        
        try:
            # Load models once (optimized)
            self._load_models()
            
            # Load data efficiently to GPU
            batch_imgs1 = self._load_images_to_gpu(img1_paths)
            batch_imgs2 = self._load_images_to_gpu(img2_paths)
            batch_points = self._load_points_from_files(points_paths)
            batch_edges = self._load_edges_from_files(edges_paths, img1_paths)
            
            # Create temporary directory for matches files
            temp_dir = tempfile.mkdtemp()
            
            # RAFT inference (optimized, no debug output)
            matches_list, matches_paths = self._batch_raft_inference(batch_imgs1, batch_imgs2, batch_points, temp_dir)
            
            # Complete InterpoNet batch inference
            success = self._batch_interponet_inference(batch_imgs1, batch_imgs2, matches_list, batch_edges, output_paths, matches_paths)
            
            # Efficient cleanup
            del batch_imgs1, batch_imgs2, batch_points, batch_edges, matches_list
            torch.cuda.empty_cache()
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return {
                'status': 'success' if success else 'error',
                'batch_size': batch_size,
                'message': f'Complete pipeline processed {batch_size} images' if success else 'Pipeline processing failed'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'batch_size': batch_size,
                'message': f'Pipeline failed: {str(e)}'
            }
        
    def handle_legacy_single_request(self, request_data):
        
        try:
            img1_path = request_data['img1_path']
            img2_path = request_data['img2_path']
            edges_path = request_data['edges_path']
            matches_path = request_data['matches_path']
            output_path = request_data['output_path']
            
            img = cv2.imread(img1_path)
            img_height, img_width = img.shape[:2]
            
            run_interponet_inference(
                img1_filename=img1_path,
                img2_filename=img2_path,
                edges_filename=edges_path,
                matches_filename=matches_path,
                out_filename=output_path,
                model_filename=self.interponet_model_path,
                ba_matches_filename=None,
                img_width=img_width,
                img_height=img_height,
                downscale=8,
                dataset_type='kitti'
            )
            
            return {
                'status': 'success',
                'output_path': output_path,
                'message': 'Legacy single frame processed'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _emergency_cleanup(self):
        try:
            if hasattr(self, 'raft_manager'):
                self.raft_manager.cleanup()
            
            if hasattr(self, 'interponet_manager'):
                self.interponet_manager.cleanup()
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
        except Exception as e:
            pass

    def _periodic_cleanup(self):
        self.request_count += 1
        
        if self.request_count % self.cleanup_interval == 0:
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                if memory_used > 6.0:
                    torch.cuda.empty_cache()

    def handle_request(self, request_data):
        try:
            self._periodic_cleanup()
            
            if request_data.get('type') == 'unified_file_pipeline':
                return self.handle_unified_file_pipeline(request_data)
            else:
                return self.handle_legacy_single_request(request_data)
                
        except Exception as e:
            self._emergency_cleanup()
            return {'status': 'error', 'message': f'Request handling failed: {str(e)}'}

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)

        while True:
            client_socket = None
            try:
                client_socket, address = server_socket.accept()

                data = b''
                while True:
                    chunk = client_socket.recv(16384)
                    if not chunk:
                        break
                    data += chunk
                    if b'\n' in data:
                        break

                if not data:
                    continue

                try:
                    request = json.loads(data.decode('utf-8').strip())
                except json.JSONDecodeError as e:
                    error_response = {'status': 'error', 'message': 'Invalid JSON format'}
                    client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    client_socket.close()
                    continue

                response = self.handle_request(request)

                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()

            except Exception as e:
                if client_socket is not None:
                    try:
                        error_response = {'status': 'error', 'message': str(e)}
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    except:
                        pass
                    try:
                        client_socket.close()
                    except:
                        pass
                
                self._emergency_cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Unified GPU-Accelerated TCP Server')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    parser.add_argument('--raft_model', type=str, default='/app/external/RAFT/models/raft-kitti.pth')
    parser.add_argument('--interponet_model', type=str, default='/app/external/InterpoNet_Pytorch/checkpoints_pytorch/converted_from_tf.pth')
    
    args = parser.parse_args()
    
    server = UnifiedTCPServer(port=args.port)
    if args.raft_model:
        server.raft_model_path = args.raft_model
    if args.interponet_model:
        server.interponet_model_path = args.interponet_model
    
    server.start()