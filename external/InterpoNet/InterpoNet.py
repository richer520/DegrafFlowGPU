import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import cv2
import time
import argparse
import gc
import os
from concurrent.futures import ThreadPoolExecutor
# Global worker function for multi-process variational processing
def variational_worker(task):
    """Worker function for multi-process variational processing"""
    import utils
    img1_path, img2_path, flo_path, out_path, dataset = task
    return utils.calc_variational_inference_map(
        img1_path, img2_path, flo_path, out_path, dataset
    )

# Global Model Manager - Singleton Mode
class InterpoNetModelManager:
    _instance = None
    _model_loaded = False
    _session = None
    _placeholders = None
    _forward_model = None
    _model_filename = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InterpoNetModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_filename):
        """Load the model (only once when needed)"""
        if self._model_loaded and self._model_filename == model_filename and self._session is not None:
            return self._session, self._placeholders, self._forward_model
            
        # Clean up existing sessions
        self.cleanup()
        
        print(f"[MODEL] Loading InterpoNet model: {model_filename}")
        
        # Reset the default image
        tf.reset_default_graph()
        
        with tf.device('/gpu:0'):
            with tf.Graph().as_default() as graph:
                # Create placeholders
                self._placeholders = {
                    'image_ph': tf.placeholder(tf.float32, shape=(None, None, None, 2), name='image_ph'),
                    'mask_ph': tf.placeholder(tf.float32, shape=(None, None, None, 1), name='mask_ph'),
                    'edges_ph': tf.placeholder(tf.float32, shape=(None, None, None, 1), name='edges_ph')
                }
                
                # Create a forward model
                self._forward_model = model.getNetwork(
                    self._placeholders['image_ph'], 
                    self._placeholders['mask_ph'], 
                    self._placeholders['edges_ph'], 
                    reuse=False
                )
                
                # Create a saver
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
                
                # Strict GPU memory configuration
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.15 
                config.allow_soft_placement = True
                config.log_device_placement = False
                
                # Create a session
                self._session = tf.Session(config=config, graph=graph)
                
               
                saver.restore(self._session, model_filename)
                
                self._model_loaded = True
                self._model_filename = model_filename
                
                print(f"[MODEL] InterpoNet model loaded successfully")
                
        return self._session, self._placeholders, self._forward_model
    
    def cleanup(self):
        
        if self._session is not None:
            try:
                self._session.close()
            except:
                pass
            self._session = None
            
        self._placeholders = None
        self._forward_model = None
        self._model_loaded = False
        self._model_filename = None
        
        # Force cleanup of TensorFlow resources
        try:
            tf.keras.backend.clear_session()
        except:
            pass
            
        try:
            tf.reset_default_graph()
        except:
            pass
            
        # Force garbage collection
        gc.collect()
    
    def get_memory_usage(self):
        """Get GPU memory usage"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                used, total = result.stdout.strip().split(', ')
                return int(used), int(total)
        except:
            pass
        return None, None

_model_manager = InterpoNetModelManager()

def run_interponet_inference(img1_filename, img2_filename, edges_filename, matches_filename,
                              out_filename, model_filename, ba_matches_filename=None,
                              img_width=1242, img_height=375, downscale=8, dataset_type='kitti'):

    edges = io_utils.load_edges_file(edges_filename, width=img_width, height=img_height)
    img, mask = io_utils.load_matching_file(matches_filename, width=img_width, height=img_height)

    print("Downscaling...")
    img, mask, edges = utils.downscale_all(img, mask, edges, downscale)

    if ba_matches_filename is not None:
        img_ba, mask_ba = io_utils.load_matching_file(ba_matches_filename, width=img_width, height=img_height)
        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, downscale)
        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, downscale)

    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            image_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 2), name='image_ph')
            mask_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='mask_ph')
            edges_ph = tf.placeholder(tf.float32, shape=(None, img.shape[0], img.shape[1], 1), name='edges_ph')

            forward_model = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

            sess = tf.Session()
            saver.restore(sess, model_filename)

            print("Performing inference...")
            prediction = sess.run(forward_model,
                                  feed_dict={
                                      image_ph: np.expand_dims(img, axis=0),
                                      mask_ph: np.reshape(mask, [1, mask.shape[0], mask.shape[1], 1]),
                                      edges_ph: np.expand_dims(np.expand_dims(edges, axis=0), axis=3),
                                  })

            print("Upscaling...")
            upscaled_pred = sk.transform.resize(prediction[0],
                                                [img_height, img_width, 2],
                                                preserve_range=True,
                                                order=3)

            print("=== Saving .flo ===")
            print("upscaled_pred shape:", upscaled_pred.shape)
            print("contains NaN:", np.isnan(upscaled_pred).any())

            io_utils.save_flow_file(upscaled_pred, filename=out_filename)


            print(f"Saved flow to: {out_filename}")


def run_interponet_inference_batch(batch_data, model_filename, 
                                   original_sizes, 
                                   downscale=8, dataset_type='kitti'):
    
    print(f"\n[BATCH] Processing {len(batch_data)} image pairs...")
    batch_start = time.time()

    try:
        mem_used_before, mem_total = _model_manager.get_memory_usage()
        if mem_used_before:
            print(f"[BATCH] GPU memory before: {mem_used_before}MB / {mem_total}MB")

        # Group by size
        print("[BATCH] Grouping images by size...")
        size_groups = {}
        for i, item in enumerate(batch_data):
            width, height = original_sizes[i]
            size_key = (width, height)
            
            if size_key not in size_groups:
                size_groups[size_key] = []
            
            size_groups[size_key].append({
                'index': i,
                'item': item,
                'size': (width, height)
            })

        for size_key, group in size_groups.items():
            print(f"  Size {size_key}: {len(group)} images")

        sess, placeholders, forward_model = _model_manager.load_model(model_filename)
        
        inference_start = time.time()
        
        # Process by group
        for size_key, group in size_groups.items():
            width, height = size_key
            group_size = len(group)
            
            # Dynamically adjust batch_size based on available memory
            available_mem = 11200 - (mem_used_before if mem_used_before else 2000)  
            max_batch_size = min(8, max(1, available_mem // 500)) 
            batch_size = min(max_batch_size, group_size)
            
            print(f"[BATCH] Processing {group_size} images of size {size_key} with batch_size={batch_size}")
            
            # batch process the current size group
            for start_idx in range(0, group_size, batch_size):
                end_idx = min(start_idx + batch_size, group_size)
                current_batch = group[start_idx:end_idx]
                
                # Add before the batch data loading loop
                data_load_start = time.time()

                batch_imgs = []
                batch_masks = []
                batch_edges_list = []
                
                for group_item in current_batch:
                    item = group_item['item']
                    
                   # Single data loading timing
                    single_load_start = time.time()
                    
                    edges = io_utils.load_edges_file(item['edges_path'], width=width, height=height)
                    img, mask = io_utils.load_matching_file(item['matches_path'], width=width, height=height)
                    
                    img, mask, edges = utils.downscale_all(img, mask, edges, downscale)
                    
                    single_load_time = time.time() - single_load_start
                    print(f"[BATCH-LOAD] Single image load took: {single_load_time:.3f}s")

                    if item.get('ba_matches_path'):
                        ba_start = time.time()
                        img_ba, mask_ba = io_utils.load_matching_file(item['ba_matches_path'], width=width, height=height)
                        img_ba, mask_ba, _ = utils.downscale_all(img_ba, mask_ba, None, downscale)
                        img, mask = utils.create_mean_map_ab_ba(img, mask, img_ba, mask_ba, downscale)
                        ba_time = time.time() - ba_start
                        print(f"[BATCH-LOAD] BA processing took: {ba_time:.3f}s")
                    
                    batch_imgs.append(img)
                    batch_masks.append(mask.reshape(mask.shape[0], mask.shape[1], 1))
                    batch_edges_list.append(np.expand_dims(edges, axis=2))
                
                data_load_time = time.time() - data_load_start
                print(f"[BATCH-LOAD] Total data loading took: {data_load_time:.3f}s")

                array_convert_start = time.time()
                batch_imgs_np = np.array(batch_imgs)
                batch_masks_np = np.array(batch_masks)  
                batch_edges_np = np.array(batch_edges_list)
                array_convert_time = time.time() - array_convert_start
                print(f"[BATCH-CONVERT] Array conversion took: {array_convert_time:.3f}s")
                
                # Batch Inference
                batch_imgs_np = np.array(batch_imgs)
                batch_masks_np = np.array(batch_masks)  
                batch_edges_np = np.array(batch_edges_list)
                
                try:
                    data_prep_end = time.time()

                    tf_inference_start = time.time()
                    
                    predictions = sess.run(forward_model,
                                        feed_dict={
                                            placeholders['image_ph']: batch_imgs_np,
                                            placeholders['mask_ph']: batch_masks_np,
                                            placeholders['edges_ph']: batch_edges_np,
                                        })
                    
                    tf_inference_time = time.time() - tf_inference_start

                except tf.errors.ResourceExhaustedError as e:
                    print(f"[BATCH] GPU memory exhausted, reducing batch size and retrying...")
                    # Downgrade to single-image processing
                    predictions = []
                    for i in range(len(current_batch)):
                        single_pred = sess.run(forward_model,
                                            feed_dict={
                                                placeholders['image_ph']: np.expand_dims(batch_imgs_np[i], 0),
                                                placeholders['mask_ph']: np.expand_dims(batch_masks_np[i], 0),
                                                placeholders['edges_ph']: np.expand_dims(batch_edges_np[i], 0),
                                            })
                        predictions.append(single_pred[0])
                    predictions = np.array(predictions)
                
                # Parallel processing results
                print(f"[BATCH] Starting parallel post-processing for {len(current_batch)} images...")
                post_process_start = time.time()

                upscaled_preds = []

                for i, group_item in enumerate(current_batch):
                    pred = predictions[i]
                    upscaled_pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_CUBIC)
                    upscaled_preds.append(upscaled_pred)

                resize_time = time.time() - post_process_start
                print(f"[BATCH] Batch resize took: {resize_time:.3f}s")
                print(f"[BATCH-RESIZE] Average resize time per image: {resize_time/len(current_batch):.3f}s")

                save_start = time.time()
                for i, group_item in enumerate(current_batch):
                    item = group_item['item'] 
                    io_utils.save_flow_file(upscaled_preds[i], filename=item['output_path'])

                save_time = time.time() - save_start
                print(f"[BATCH] Batch save took: {save_time:.3f}s")
                print(f"[BATCH-SAVE] Average save time per image: {save_time/len(current_batch):.3f}s")

                # Parallel execution of variational processing
                variational_tasks = []
                for i, group_item in enumerate(current_batch):
                    item = group_item['item']
                    if item.get('apply_variational', True):
                        variational_tasks.append((
                            item['img1_path'], item['img2_path'],
                            item['output_path'], item['output_path'],
                            dataset_type
                        ))

                if variational_tasks:
                    print(f"[BATCH] Starting parallel variational processing for {len(variational_tasks)} images...")
                    cpu_count = os.cpu_count()
                    num_processes = min(len(variational_tasks), 6)
                    print(f"[BATCH] System has {cpu_count} CPU cores, using {num_processes} processes")
                    print(f"[BATCH] Using {num_processes} parallel processes")
                    
                    print(f"[BATCH] Using ThreadPoolExecutor with {num_processes} threads")
                    with ThreadPoolExecutor(max_workers=num_processes) as executor:
                        list(executor.map(variational_worker, variational_tasks))
                    


                total_post_time = time.time() - post_process_start
                print(f"[BATCH] Total post-processing took: {total_post_time:.3f}s")
                
                # Force cleanup of intermediate variables
                del batch_imgs_np, batch_masks_np, batch_edges_np, predictions
                gc.collect()
        
        inference_time = time.time() - inference_start
        print(f"[BATCH] GPU inference completed in {inference_time:.2f}s")

        # Check memory usage
        mem_used_after, _ = _model_manager.get_memory_usage()
        if mem_used_before and mem_used_after:
            mem_increase = mem_used_after - mem_used_before
            print(f"[BATCH] GPU memory after: {mem_used_after}MB (increase: +{mem_increase}MB)")
            
            if mem_increase > 1000: 
                _model_manager.cleanup()

    except Exception as e:
        _model_manager.cleanup()
        raise e
    
    total_time = time.time() - batch_start
    avg_time = total_time / len(batch_data)
    
    print(f"\n[BATCH] ===== BATCH SUMMARY =====")
    print(f"[BATCH] Total images: {len(batch_data)}")
    print(f"[BATCH] Total time: {total_time:.2f}s")
    print(f"[BATCH] Average per image: {avg_time:.3f}s")
    print(f"[BATCH] GPU inference time: {inference_time:.2f}s")
    print(f"[BATCH] Speedup: {(len(batch_data) * 1.441) / total_time:.2f}x") 
    print(f"[BATCH] =======================\n")
    
    return True

def force_cleanup():
    """Force cleanup of external interfaces for all resources"""
    global _model_manager
    _model_manager.cleanup()