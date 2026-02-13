#!/usr/bin/env python3
"""
Enhanced TCP server - supports single frame and batch processing
"""

import socket
import json
import time
import os
import sys
import skimage.io
import numpy as np
import gc
import tensorflow as tf

sys.path.append('/app/external/InterpoNet')
sys.path.append('/app')

# Import single frame and batch processing functions
from InterpoNet import run_interponet_inference, run_interponet_inference_batch, force_cleanup


class TCPServer:
    def __init__(self, port=9999):
        self.port = port
        self.model_path = '/app/external/InterpoNet/models/best_model_kitti2015.ckpt'
        self.request_count = 0
        self.cleanup_interval = 10  
        
    def get_gpu_memory_info(self):
        """Get GPU memory information"""
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
        """Record memory status"""
        used, total, free = self.get_gpu_memory_info()
        if used is not None:
            usage_percent = (used / total) * 100
            print(f"[MEMORY{stage}] GPU: {used}MB/{total}MB used ({usage_percent:.1f}%), {free}MB free")
        else:
            print(f"[MEMORY{stage}] GPU memory info unavailable")

    def handle_single_request(self, request_data):
        """Process a single request"""
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

            # Read image size
            img = skimage.io.imread(img1_path)
            img_height, img_width = img.shape[:2]
            downscale = request_data.get('downscale', 8)
            dataset_type = request_data.get('dataset', 'kitti')

            # Make sure the output directory exists
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            #Call single frame processing
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
            
            
            self._emergency_cleanup()
            
            return {
                'status': 'error',
                'message': str(e)
            }

    def handle_batch_request(self, request_data):
        """Processing batch requests - Enhanced memory management"""
        try:
            print(f"[BATCH] Processing batch request...")
            self.log_memory_status(" BEFORE")
            
            start_time = time.time()
            
            batch_size = request_data['batch_size']
            print(f"[BATCH] Received batch request for {batch_size} image pairs")
            
            # Check if GPU memory is sufficient
            used, total, free = self.get_gpu_memory_info()
            if used is not None and free < 2000: 
                print(f"[BATCH] WARNING: Low GPU memory ({free}MB free), forcing cleanup...")
                self._emergency_cleanup()
                time.sleep(1)
            
            # Extract batch data
            img1_paths = request_data['img1_paths']
            img2_paths = request_data['img2_paths']
            edges_paths = request_data['edges_paths']
            matches_paths = request_data['matches_paths']
            output_paths = request_data['output_paths']
            downscale = request_data.get('downscale', 8)
            dataset_type = request_data.get('dataset', 'kitti')
            
            # Make sure all output directories exist
            for output_path in output_paths:
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
            
            # Prepare batch data
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
            
            # Get the individual dimensions of each image
            print(f"[BATCH] Reading image dimensions...")
            original_sizes = []
            for item in batch_data:
                img = skimage.io.imread(item['img1_path'])
                original_sizes.append((img.shape[1], img.shape[0]))
                del img  
            
            # Call batch processing
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
            
            
            self._emergency_cleanup()
            
            return {
                'status': 'error',
                'message': str(e)
            }

    def _emergency_cleanup(self):
        """Urgent clean up all resources"""
        try:
            print("[CLEANUP] Performing emergency cleanup...")
            
            # Call InterpoNet's forced cleanup
            force_cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Try to clean up TensorFlow resources
            try:
                tf.keras.backend.clear_session()
                print("[CLEANUP] TensorFlow cleanup completed")
            except Exception as tf_error:
                print(f"[CLEANUP] TensorFlow cleanup failed: {tf_error}")
            
            print("[CLEANUP] Emergency cleanup completed")
            
        except Exception as cleanup_error:
            print(f"[CLEANUP] Emergency cleanup failed: {cleanup_error}")

    def _periodic_cleanup(self):
        """Regular cleaning mechanism"""
        self.request_count += 1
        
        if self.request_count % self.cleanup_interval == 0:
            print(f"[CLEANUP] Periodic cleanup after {self.request_count} requests...")
            
            # Check memory usage
            used, total, free = self.get_gpu_memory_info()
            if used is not None:
                usage_percent = (used / total) * 100
                if usage_percent > 70:  
                    print(f"[CLEANUP] High memory usage ({usage_percent:.1f}%), forcing cleanup...")
                    self._emergency_cleanup()
                else:
                    print(f"[CLEANUP] Memory usage normal ({usage_percent:.1f}%), skipping cleanup")

    def handle_request(self, request_data):
        """Distribute processing according to request type"""
        try:
            # Perform periodic cleanup checks
            self._periodic_cleanup()
            
            # Check if it is batch mode
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
        """Start TCP server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase receive buffer size to handle batch requests
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
        
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)

        print(f"[INFO] Enhanced TCP Server (with memory management) listening on port {self.port}")
        print(f"[INFO] Model path: {self.model_path}")
        print(f"[INFO] Cleanup interval: every {self.cleanup_interval} requests")
        print("[INFO] Ready to accept connections...")
        
        # Record memory status at startup
        self.log_memory_status(" STARTUP")

        while True:
            client_socket = None
            try:
                client_socket, address = server_socket.accept()
                print(f"\n[INFO] Connection from {address}")

                # Receive potentially larger batches of requests
                data = b''
                while True:
                    chunk = client_socket.recv(8192)  
                    if not chunk:
                        break
                    data += chunk
                    if b'\n' in data:
                        break

                if not data:
                    print("[WARN] Empty request received")
                    continue

                # Parsing JSON Requests
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

                # Execute processing
                response = self.handle_request(request)

                # Return a response
                response_data = (json.dumps(response) + '\n').encode('utf-8')
                client_socket.send(response_data)
                client_socket.close()

                print(f"[INFO] Request completed: {response['status']}")
                
                # Record memory status after request is completed
                if self.request_count % 5 == 0: 
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
                
                self._emergency_cleanup()


if __name__ == "__main__":
    server = TCPServer(port=9999)
    server.start()