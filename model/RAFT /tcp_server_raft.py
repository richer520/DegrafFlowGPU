#!/usr/bin/env python3
"""
RAFT TCP服务器 - 负责网络通信，调用 degraf_raft_matcher.py 中的 run_raft_matching()
提供长连接服务，避免重复加载模型
"""

import socket
import json
import time
import os
import sys

# 添加RAFT路径
sys.path.append('/app/external/RAFT')
sys.path.append('/app/external/RAFT/core')
sys.path.append('/app')

# 从 degraf_raft_matcher.py 导入封装好的函数
from degraf_raft_matcher import run_raft_matching


class RAFTTCPServer:
    def __init__(self, port=9998):
        """
        初始化RAFT TCP服务器
        Args:
            port: 监听端口（默认9998，避免与InterpoNet的9999冲突）
        """
        self.port = port
        self.model_path = '/app/external/RAFT/models/raft-kitti.pth'
        
    def handle_request(self, request_data):
        """处理单个请求"""
        try:
            start_time = time.time()
            
            # 提取请求参数
            image1_path = request_data['image1_path']
            image2_path = request_data['image2_path']
            points_path = request_data['points_path']
            output_path = request_data['output_path']
            
            # 可选参数
            model_path = request_data.get('model_path', self.model_path)
            small = request_data.get('small', False)
            mixed_precision = request_data.get('mixed_precision', False)
            alternate_corr = request_data.get('alternate_corr', False)
            
            # 验证输入文件存在
            if not os.path.exists(image1_path):
                raise FileNotFoundError(f"Image1 not found: {image1_path}")
            if not os.path.exists(image2_path):
                raise FileNotFoundError(f"Image2 not found: {image2_path}")
            if not os.path.exists(points_path):
                raise FileNotFoundError(f"Points file not found: {points_path}")
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 调用RAFT匹配函数
            num_matches = run_raft_matching(
                image1_path=image1_path,
                image2_path=image2_path,
                points_path=points_path,
                output_path=output_path,
                model_path=model_path,
                small=small,
                mixed_precision=mixed_precision,
                alternate_corr=alternate_corr
            )
            
            total_time = time.time() - start_time
            
            return {
                'status': 'success',
                'output_path': output_path,
                'num_matches': num_matches,
                'processing_time': total_time,
                'message': f'Processed {num_matches} matches in {total_time:.2f}s'
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start(self):
        """启动TCP服务器"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        
        print(f"[INFO] RAFT TCP Server listening on port {self.port}")
        print(f"[INFO] Model path: {self.model_path}")
        print("[INFO] Ready to accept connections...")
        
        # 预加载模型（第一次调用时会加载）
        print("[INFO] Pre-loading RAFT model...")
        from degraf_raft_matcher import load_model
        load_model(self.model_path)
        print("[INFO] Model loaded and cached in memory")
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                print(f"\n[INFO] Connection from {address}")
                
                # 接收数据
                data = b''
                while True:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    # 检查是否接收完成（简单协议：JSON以换行符结束）
                    if b'\n' in data:
                        break
                
                # 解析JSON请求
                request_str = data.decode('utf-8').strip()
                request = json.loads(request_str)
                
                print(f"[INFO] Processing: {request.get('image1_path', 'unknown')} -> {request.get('image2_path', 'unknown')}")
                
                # 执行处理
                response = self.handle_request(request)
                
                # 返回响应
                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()
                
                print(f"[INFO] Request completed: {response['status']}")
                if response['status'] == 'success':
                    print(f"[INFO] Matches saved to: {response['output_path']}")
                
            except KeyboardInterrupt:
                print("\n[INFO] Server shutting down...")
                break
            except Exception as e:
                print(f"[ERROR] Server error: {e}")
                if 'client_socket' in locals():
                    try:
                        error_response = {'status': 'error', 'message': str(e)}
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    except:
                        pass
                    client_socket.close()
        
        server_socket.close()
        print("[INFO] Server stopped")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='RAFT TCP Server')
    parser.add_argument('--port', type=int, default=9998, help='Server port (default: 9998)')
    parser.add_argument('--model', type=str, default='/app/models/raft-kitti.pth', 
                        help='Path to RAFT model checkpoint')
    args = parser.parse_args()
    
    server = RAFTTCPServer(port=args.port)
    if args.model:
        server.model_path = args.model
    server.start()


if __name__ == "__main__":
    main()