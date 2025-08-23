#!/usr/bin/env python3
"""
增强的TCP服务器 - 支持单帧和批量处理
"""

import socket
import json
import time
import os
import sys
import skimage.io
import numpy as np
import gc

sys.path.append('/app/external/InterpoNet')
sys.path.append('/app')

# 导入单帧和批量处理函数
from InterpoNet import run_interponet_inference, run_interponet_inference_batch, force_cleanup


class TCPServer:
    def __init__(self, port=9999):
        self.port = port
        self.model_path = '/app/external/InterpoNet/checkpoints_improved/best_model_epoch_1.ckpt'
        self.request_count = 0
        self.cleanup_interval = 10  # 每10次请求后强制清理
        
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', 
                                   '--format=csv,nounits,noheader'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, timeout=5)
            if result.returncode == 0:
                used, total, free = result.stdout.strip().split(', ')
                return int(used), int(total), int(free)
        except Exception as e:
            print(f"[DEBUG] Failed to get GPU memory info: {e}")
        return None, None, None

    def log_memory_status(self, stage=""):
        """记录内存状态"""
        used, total, free = self.get_gpu_memory_info()
        if used is not None:
            usage_percent = (used / total) * 100
            print(f"[MEMORY{stage}] GPU: {used}MB/{total}MB used ({usage_percent:.1f}%), {free}MB free")
        else:
            print(f"[MEMORY{stage}] GPU memory info unavailable")

    def handle_single_request(self, request_data):
        """处理单个请求（保持原有逻辑）"""
        try:
            print(f"[SINGLE] Processing single frame request...")
            self.log_memory_status(" BEFORE")
            
            start_time = time.time()
            
            img1_path = request_data['img1_path']
            img2_path = request_data['img2_path']
            edges_path = request_data['edges_path']
            matches_path = request_data['matches_path']
            output_path = request_data['output_path']
            ba_matches_path = request_data.get('ba_matches_path', None)

            # 读取图像尺寸
            img = skimage.io.imread(img1_path)
            img_height, img_width = img.shape[:2]
            downscale = request_data.get('downscale', 8)
            dataset_type = request_data.get('dataset', 'kitti')

            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 调用单帧处理
            print(f"[DEBUG-SINGLE] About to call InterpoNet inference...")
            interponet_start = time.time()

            run_interponet_inference(
                img1_filename=img1_path,
                img2_filename=img2_path,
                edges_filename=edges_path,
                matches_filename=matches_path,
                out_filename=output_path,
                model_filename=self.model_path,
                ba_matches_filename=ba_matches_path,
                img_width=img_width,
                img_height=img_height,
                downscale=downscale,
                dataset_type=dataset_type
            )

            interponet_time = time.time() - interponet_start
            print(f"[DEBUG-SINGLE] InterpoNet inference took: {interponet_time:.3f}s")

            total_time = time.time() - start_time
            self.log_memory_status(" AFTER")

            return {
                'status': 'success',
                'output_path': output_path,
                'processing_time': total_time,
                'message': f'Processed successfully in {total_time:.2f}s'
            }

        except Exception as e:
            print(f"[ERROR] Single request failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 出错时强制清理
            self._emergency_cleanup()
            
            return {
                'status': 'error',
                'message': str(e)
            }

    def handle_batch_request(self, request_data):
        """处理批量请求 - 增强内存管理"""
        try:
            print(f"[BATCH] Processing batch request...")
            self.log_memory_status(" BEFORE")
            
            start_time = time.time()
            
            batch_size = request_data['batch_size']
            print(f"[BATCH] Received batch request for {batch_size} image pairs")
            
            # 检查GPU内存是否足够
            used, total, free = self.get_gpu_memory_info()
            if used is not None and free < 2000:  # 少于2GB可用内存
                print(f"[BATCH] WARNING: Low GPU memory ({free}MB free), forcing cleanup...")
                self._emergency_cleanup()
                time.sleep(1)  # 等待清理完成
            
            # 提取批量数据
            img1_paths = request_data['img1_paths']
            img2_paths = request_data['img2_paths']
            edges_paths = request_data['edges_paths']
            matches_paths = request_data['matches_paths']
            output_paths = request_data['output_paths']
            downscale = request_data.get('downscale', 8)
            dataset_type = request_data.get('dataset', 'kitti')
            
            # 确保所有输出目录存在
            for output_path in output_paths:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            
            # 准备批量数据
            batch_data = []
            for i in range(batch_size):
                item = {
                    'img1_path': img1_paths[i],
                    'img2_path': img2_paths[i],
                    'edges_path': edges_paths[i],
                    'matches_path': matches_paths[i],
                    'output_path': output_paths[i],
                    'ba_matches_path': None,
                    'apply_variational': True
                }
                batch_data.append(item)
            
            # 获取每张图像的各自尺寸
            print(f"[BATCH] Reading image dimensions...")
            original_sizes = []
            for item in batch_data:
                img = skimage.io.imread(item['img1_path'])
                original_sizes.append((img.shape[1], img.shape[0]))
                del img  # 及时释放
            
            # 调用批量处理
            print(f"[BATCH] Starting batch inference with {len(batch_data)} samples...")
            batch_interponet_start = time.time()

            success = run_interponet_inference_batch(
                batch_data=batch_data,
                model_filename=self.model_path,
                original_sizes=original_sizes,
                downscale=downscale,
                dataset_type=dataset_type
            )

            batch_interponet_time = time.time() - batch_interponet_start
            avg_per_image = batch_interponet_time / len(batch_data)
            print(f"[DEBUG-BATCH] Total InterpoNet batch time: {batch_interponet_time:.3f}s")
            print(f"[DEBUG-BATCH] Average per image: {avg_per_image:.3f}s")
            
            total_time = time.time() - start_time
            avg_time = total_time / batch_size
            
            self.log_memory_status(" AFTER")
            
            return {
                'status': 'success' if success else 'error',
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_image': avg_time,
                'message': f'Batch processed {batch_size} images in {total_time:.2f}s (avg: {avg_time:.3f}s/img)'
            }
            
        except Exception as e:
            print(f"[ERROR] Batch request failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 出错时强制清理
            self._emergency_cleanup()
            
            return {
                'status': 'error',
                'message': str(e)
            }

    def _emergency_cleanup(self):
        """紧急清理所有资源"""
        try:
            print("[CLEANUP] Performing emergency cleanup...")
            
            # 调用InterpoNet的强制清理
            force_cleanup()
            
            # 强制垃圾回收
            gc.collect()
            
            # 尝试清理TensorFlow资源
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                print("[CLEANUP] TensorFlow cleanup completed")
            except Exception as tf_error:
                print(f"[CLEANUP] TensorFlow cleanup failed: {tf_error}")
            
            print("[CLEANUP] Emergency cleanup completed")
            
        except Exception as cleanup_error:
            print(f"[CLEANUP] Emergency cleanup failed: {cleanup_error}")

    def _periodic_cleanup(self):
        """定期清理机制"""
        self.request_count += 1
        
        if self.request_count % self.cleanup_interval == 0:
            print(f"[CLEANUP] Periodic cleanup after {self.request_count} requests...")
            
            # 检查内存使用情况
            used, total, free = self.get_gpu_memory_info()
            if used is not None:
                usage_percent = (used / total) * 100
                if usage_percent > 70:  # 使用率超过70%
                    print(f"[CLEANUP] High memory usage ({usage_percent:.1f}%), forcing cleanup...")
                    self._emergency_cleanup()
                else:
                    print(f"[CLEANUP] Memory usage normal ({usage_percent:.1f}%), skipping cleanup")

    def handle_request(self, request_data):
        """根据请求类型分发处理"""
        try:
            # 执行定期清理检查
            self._periodic_cleanup()
            
            # 检查是否为批量模式
            if request_data.get('batch_mode', False):
                result = self.handle_batch_request(request_data)
            else:
                result = self.handle_single_request(request_data)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Request handling failed: {e}")
            self._emergency_cleanup()
            return {
                'status': 'error',
                'message': f'Request handling failed: {str(e)}'
            }

    def start(self):
        """启动TCP服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 增加接收缓冲区大小以处理批量请求
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
        
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)

        print(f"[INFO] Enhanced TCP Server (with memory management) listening on port {self.port}")
        print(f"[INFO] Model path: {self.model_path}")
        print(f"[INFO] Cleanup interval: every {self.cleanup_interval} requests")
        print("[INFO] Ready to accept connections...")
        
        # 启动时记录内存状态
        self.log_memory_status(" STARTUP")

        while True:
            client_socket = None
            try:
                client_socket, address = server_socket.accept()
                print(f"\n[INFO] Connection from {address}")

                # 接收可能更大的批量请求
                data = b''
                while True:
                    chunk = client_socket.recv(8192)  # 增加块大小
                    if not chunk:
                        break
                    data += chunk
                    # 检查是否收到完整的JSON（以换行符结束）
                    if b'\n' in data:
                        break

                if not data:
                    print("[WARN] Empty request received")
                    continue

                # 解析 JSON 请求
                try:
                    request = json.loads(data.decode('utf-8').strip())
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Invalid JSON request: {e}")
                    error_response = {'status': 'error', 'message': 'Invalid JSON format'}
                    client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    client_socket.close()
                    continue
                
                if request.get('batch_mode', False):
                    print(f"[INFO] Processing BATCH request for {request.get('batch_size', 0)} images")
                else:
                    print(f"[INFO] Processing SINGLE request for: {request.get('output_path', 'unknown')}")

                # 执行处理
                response = self.handle_request(request)

                # 返回响应
                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()

                print(f"[INFO] Request completed: {response['status']}")
                
                # 请求完成后记录内存状态
                if self.request_count % 5 == 0:  # 每5个请求记录一次
                    self.log_memory_status(f" REQUEST_{self.request_count}")

            except Exception as e:
                print(f"[ERROR] Server error: {e}")
                import traceback
                traceback.print_exc()
                
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
                
                # 服务器错误时也执行清理
                self._emergency_cleanup()


if __name__ == "__main__":
    server = TCPServer(port=9999)
    server.start()