#!/usr/bin/env python3
"""
RAFT Batch TCP Server - High Performance Batch Processing
Supports batch requests from C++ FeatureMatcher::callRAFTTCP_batch()
"""

import socket
import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.environ.get(
    "DEGRAF_PROJECT_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
)

sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "core"))
sys.path.append(PROJECT_ROOT)

from degraf_raft_matcher import RAFTModelManager, run_raft_matching_batch

class RAFTBatchTCPServer:
    def __init__(self, port=9998, model_path=None):
        self.port = port
        if model_path is None:
            default_model = os.environ.get("RAFT_MODEL_PATH")
            if default_model:
                self.model_path = default_model
            else:
                self.model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "raft-kitti.pth")
        else:
            self.model_path = model_path
        self.model_manager = RAFTModelManager()
        
    def handle_batch_request(self, request_data):
        try:
            batch_size = request_data['batch_size']
            img1_paths = request_data['image1_paths']
            img2_paths = request_data['image2_paths']
            points_paths = request_data['points_paths']
            output_paths = request_data['output_paths']
            
            # Ensure output directory exists
            for output_path in output_paths:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            
            batch_data = []
            for i in range(batch_size):
                batch_data.append({
                    'image1_path': img1_paths[i],
                    'image2_path': img2_paths[i],
                    'points_path': points_paths[i],
                    'output_path': output_paths[i]
                })
            
            num_processed = run_raft_matching_batch(
                batch_data=batch_data,
                model_path=self.model_path
            )
            
            return {
                'status': 'success',
                'batch_size': batch_size,
                'processed': num_processed,
                'message': f'Processed {num_processed} image pairs successfully'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        
        # Pre-load model
        self.model_manager.load_model(self.model_path)
        
        while True:
            try:
                client_socket, address = server_socket.accept()
                
                # Receive data
                data = b''
                while True:
                    chunk = client_socket.recv(8192)
                    if not chunk:
                        break
                    data += chunk
                    if b'\n' in data:
                        break
                
                # Parse JSON request
                request_str = data.decode('utf-8').strip()
                request = json.loads(request_str)
                
                # Handle batch request
                if request.get('batch_mode', False):
                    response = self.handle_batch_request(request)
                else:
                    response = {'status': 'error', 'message': 'Only batch mode supported'}
                
                # Send response
                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                if 'client_socket' in locals():
                    try:
                        error_response = {'status': 'error', 'message': str(e)}
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
                    except:
                        pass
                    client_socket.close()
        
        server_socket.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RAFT Batch TCP Server')
    parser.add_argument('--port', type=int, default=9998, help='Server port')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to RAFT model checkpoint')
    args = parser.parse_args()
    
    server = RAFTBatchTCPServer(port=args.port, model_path=args.model)
    server.start()

if __name__ == "__main__":
    main()