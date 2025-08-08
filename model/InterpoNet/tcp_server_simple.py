#!/usr/bin/env python3
"""
简单的TCP服务器 - 只负责网络通信，调用 InterpoNet.py 中的 run_interponet_inference()
"""

import socket
import json
import time
import os
import sys
import skimage.io
# 添加路径以导入 InterpoNet.py
sys.path.append('/app/external/InterpoNet')  # 根据你的目录结构自行修改
sys.path.append('/app')

# ✅ 从 InterpoNet.py 中导入封装好的函数
from InterpoNet import run_interponet_inference


class TCPServer:
    def __init__(self, port=9999):
        self.port = port

    def handle_request(self, request_data):
        """处理单个请求"""
        try:
            start_time = time.time()
            # 提取请求参数
            img1_path = request_data['img1_path']
            img2_path = request_data['img2_path']
            edges_path = request_data['edges_path']
            matches_path = request_data['matches_path']
            output_path = request_data['output_path']
            ba_matches_path = request_data.get('ba_matches_path', None)

            # 模型与参数设置
            model_path = '/app/external/InterpoNet/checkpoints/fine_tuned_kitti2015.ckpt'
            img = skimage.io.imread(img1_path)
            img_height, img_width = img.shape[:2]
            # img_width = request_data.get('img_width', 1242)
            # img_height = request_data.get('img_height', 375)
            downscale = request_data.get('downscale', 8)
            dataset_type = request_data.get('dataset', 'kitti')

            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # ✅ 调用封装函数进行推理
            run_interponet_inference(
                img1_filename=img1_path,
                img2_filename=img2_path,
                edges_filename=edges_path,
                matches_filename=matches_path,
                out_filename=output_path,
                model_filename=model_path,
                ba_matches_filename=ba_matches_path,
                img_width=img_width,
                img_height=img_height,
                downscale=downscale,
                dataset_type=dataset_type
            )

            total_time = time.time() - start_time

            return {
                'status': 'success',
                'output_path': output_path,
                'processing_time': total_time,
                'message': f'Processed successfully in {total_time:.2f}s'
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

        print(f"[INFO] TCP Server listening on port {self.port}")
        print("[INFO] Ready to accept connections...")

        while True:
            try:
                client_socket, address = server_socket.accept()
                print(f"\n[INFO] Connection from {address}")

                data = b''
                while True:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if len(chunk) < 4096:
                        break

                # 解析 JSON 请求
                request = json.loads(data.decode('utf-8'))
                print(f"[INFO] Processing request for: {request.get('output_path', 'unknown')}")

                # 执行处理
                response = self.handle_request(request)

                # 返回响应
                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()

                print(f"[INFO] Request completed: {response['status']}")

            except Exception as e:
                print(f"[ERROR] Server error: {e}")
                if 'client_socket' in locals():
                    try:
                        error_response = {'status': 'error', 'message': str(e)}
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    except:
                        pass
                    client_socket.close()


if __name__ == "__main__":
    server = TCPServer(port=9999)
    server.start()
