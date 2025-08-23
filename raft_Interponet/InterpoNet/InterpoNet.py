import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import cv2
import argparse
import gc

# 全局worker函数用于多进程variational处理
def variational_worker(task):
    """多进程variational处理的worker函数"""
    import utils
    img1_path, img2_path, flo_path, out_path, dataset = task
    return utils.calc_variational_inference_map(
        img1_path, img2_path, flo_path, out_path, dataset
    )

# 全局模型管理器 - 单例模式
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
        """加载模型（仅在需要时加载一次）"""
        if self._model_loaded and self._model_filename == model_filename and self._session is not None:
            return self._session, self._placeholders, self._forward_model
            
        # 清理现有会话
        self.cleanup()
        
        print(f"[MODEL] Loading InterpoNet model: {model_filename}")
        
        # 重置默认图
        tf.reset_default_graph()
        
        with tf.device('/gpu:0'):
            with tf.Graph().as_default() as graph:
                # 创建placeholders
                self._placeholders = {
                    'image_ph': tf.placeholder(tf.float32, shape=(None, None, None, 2), name='image_ph'),
                    'mask_ph': tf.placeholder(tf.float32, shape=(None, None, None, 1), name='mask_ph'),
                    'edges_ph': tf.placeholder(tf.float32, shape=(None, None, None, 1), name='edges_ph')
                }
                
                # 创建前向模型
                self._forward_model = model.getNetwork(
                    self._placeholders['image_ph'], 
                    self._placeholders['mask_ph'], 
                    self._placeholders['edges_ph'], 
                    reuse=False
                )
                
                # 创建saver
                saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
                
                # 严格的GPU内存配置
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.15  # 降低到40%
                config.allow_soft_placement = True
                config.log_device_placement = False
                
                # 创建会话
                self._session = tf.Session(config=config, graph=graph)
                
                # 恢复模型
                saver.restore(self._session, model_filename)
                
                self._model_loaded = True
                self._model_filename = model_filename
                
                print(f"[MODEL] InterpoNet model loaded successfully")
                
        return self._session, self._placeholders, self._forward_model
    
    def cleanup(self):
        """强制清理所有资源"""
        print("[MODEL] Cleaning up InterpoNet model resources...")
        
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
        
        # 强制清理TensorFlow资源
        try:
            tf.keras.backend.clear_session()
        except:
            pass
            
        try:
            tf.reset_default_graph()
        except:
            pass
            
        # 强制垃圾回收
        gc.collect()
        
        print("[MODEL] Model cleanup completed")
    
    def get_memory_usage(self):
        """获取GPU内存使用情况"""
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

# 全局模型管理器实例
_model_manager = InterpoNetModelManager()

def run_interponet_inference(img1_filename, img2_filename, edges_filename, matches_filename,
                              out_filename, model_filename, ba_matches_filename=None,
                              img_width=1242, img_height=375, downscale=8, dataset_type='kitti'):
    """封装为函数以供调用"""
    print("Loading files...")
    print("DEBUG: img_width =", img_width)
    print("DEBUG: img_height =", img_height)

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

            # utils.calc_variational_inference_map(img1_filename, img2_filename,
            #                                      out_filename, out_filename,
            #                                      dataset_type)

            print(f"Saved flow to: {out_filename}")


def run_interponet_inference_batch(batch_data, model_filename, 
                                   original_sizes, 
                                   downscale=8, dataset_type='kitti'):
    """批量处理版本 - 修复内存泄漏版本"""
    import time
    
    print(f"\n[BATCH] Processing {len(batch_data)} image pairs...")
    batch_start = time.time()

    try:
        # 获取内存使用情况
        mem_used_before, mem_total = _model_manager.get_memory_usage()
        if mem_used_before:
            print(f"[BATCH] GPU memory before: {mem_used_before}MB / {mem_total}MB")

        # 按尺寸分组
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

        print(f"[BATCH] Found {len(size_groups)} different image sizes:")
        for size_key, group in size_groups.items():
            print(f"  Size {size_key}: {len(group)} images")

        # 加载模型（复用机制）
        print("[BATCH] Loading/reusing model...")
        sess, placeholders, forward_model = _model_manager.load_model(model_filename)
        
        print("[BATCH] Processing groups...")
        inference_start = time.time()
        
        # 按组处理
        for size_key, group in size_groups.items():
            width, height = size_key
            group_size = len(group)
            
            # 根据可用内存动态调整batch_size
            available_mem = 11200 - (mem_used_before if mem_used_before else 2000)  # 估算可用内存
            max_batch_size = min(8, max(1, available_mem // 500))  # 每500MB处理1张图
            batch_size = min(max_batch_size, group_size)
            
            print(f"[BATCH] Processing {group_size} images of size {size_key} with batch_size={batch_size}")
            
            # 分批处理当前尺寸组
            for start_idx in range(0, group_size, batch_size):
                end_idx = min(start_idx + batch_size, group_size)
                current_batch = group[start_idx:end_idx]
                
                # 在批量数据加载循环前添加
                data_load_start = time.time()

                # 加载当前批次数据
                batch_imgs = []
                batch_masks = []
                batch_edges_list = []
                
                for group_item in current_batch:
                    item = group_item['item']
                    
                    # 单个数据加载计时
                    single_load_start = time.time()
                    
                    edges = io_utils.load_edges_file(item['edges_path'], width=width, height=height)
                    img, mask = io_utils.load_matching_file(item['matches_path'], width=width, height=height)
                    
                    # 立即进行downscale，避免处理大图像
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

                # 数组转换计时
                array_convert_start = time.time()
                batch_imgs_np = np.array(batch_imgs)
                batch_masks_np = np.array(batch_masks)  
                batch_edges_np = np.array(batch_edges_list)
                array_convert_time = time.time() - array_convert_start
                print(f"[BATCH-CONVERT] Array conversion took: {array_convert_time:.3f}s")
                
                # 批量推理
                batch_imgs_np = np.array(batch_imgs)
                batch_masks_np = np.array(batch_masks)  
                batch_edges_np = np.array(batch_edges_list)
                
                try:
                    print(f"[BATCH-TF] About to run TF inference for {len(batch_imgs_np)} images")
                    print(f"[BATCH-TF] Input shapes: img={batch_imgs_np.shape}, mask={batch_masks_np.shape}, edges={batch_edges_np.shape}")
                    # 添加数据准备时间
                    data_prep_end = time.time()

                    tf_inference_start = time.time()
                    
                    predictions = sess.run(forward_model,
                                        feed_dict={
                                            placeholders['image_ph']: batch_imgs_np,
                                            placeholders['mask_ph']: batch_masks_np,
                                            placeholders['edges_ph']: batch_edges_np,
                                        })
                    
                    tf_inference_time = time.time() - tf_inference_start
                    print(f"[BATCH-TF] Pure TensorFlow inference took: {tf_inference_time:.3f}s")
                    print(f"[BATCH-TF] Per image TF time: {tf_inference_time/len(batch_imgs_np):.3f}s")
                    print(f"[BATCH-TF] Output shape: {predictions.shape}")

                except tf.errors.ResourceExhaustedError as e:
                    print(f"[BATCH] GPU memory exhausted, reducing batch size and retrying...")
                    # 降级为单张处理
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
                
                # 并行处理结果
                print(f"[BATCH] Starting parallel post-processing for {len(current_batch)} images...")
                post_process_start = time.time()

                # 批量resize (仍需循环，但可以优化)
                upscaled_preds = []

                for i, group_item in enumerate(current_batch):
                    pred = predictions[i]
                    # 使用OpenCV替代skimage进行resize（更快）
                    upscaled_pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_CUBIC)
                    upscaled_preds.append(upscaled_pred)

                resize_time = time.time() - post_process_start
                print(f"[BATCH] Batch resize took: {resize_time:.3f}s")
                print(f"[BATCH-RESIZE] Average resize time per image: {resize_time/len(current_batch):.3f}s")


                # 批量保存
                save_start = time.time()
                for i, group_item in enumerate(current_batch):
                    item = group_item['item'] 
                    io_utils.save_flow_file(upscaled_preds[i], filename=item['output_path'])

                save_time = time.time() - save_start
                print(f"[BATCH] Batch save took: {save_time:.3f}s")
                # 在现有的save计时代码后添加  
                print(f"[BATCH-SAVE] Average save time per image: {save_time/len(current_batch):.3f}s")

                # 并行执行variational处理
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
                    
                    # 使用多进程并行处理
                    from concurrent.futures import ThreadPoolExecutor
                    
                    # 创建进程池（根据CPU核心数调整）
                    # 增加并行进程数，根据CPU核心数优化
                    import os
                    cpu_count = os.cpu_count()
                    num_processes = min(len(variational_tasks), 6)
                    print(f"[BATCH] System has {cpu_count} CPU cores, using {num_processes} processes")
                    print(f"[BATCH] Using {num_processes} parallel processes")
                    
                    print(f"[BATCH] Using ThreadPoolExecutor with {num_processes} threads")
                    with ThreadPoolExecutor(max_workers=num_processes) as executor:
                        list(executor.map(variational_worker, variational_tasks))
                    
                    print(f"[BATCH] Parallel variational processing completed")


                total_post_time = time.time() - post_process_start
                print(f"[BATCH] Total post-processing took: {total_post_time:.3f}s")
                
                # 强制清理中间变量
                del batch_imgs_np, batch_masks_np, batch_edges_np, predictions
                gc.collect()
        
        inference_time = time.time() - inference_start
        print(f"[BATCH] GPU inference completed in {inference_time:.2f}s")

        # 检查内存使用情况
        mem_used_after, _ = _model_manager.get_memory_usage()
        if mem_used_before and mem_used_after:
            mem_increase = mem_used_after - mem_used_before
            print(f"[BATCH] GPU memory after: {mem_used_after}MB (increase: +{mem_increase}MB)")
            
            # 如果内存增长过多，强制清理
            if mem_increase > 1000:  # 超过1GB增长
                print(f"[BATCH] WARNING: Large memory increase detected, forcing cleanup...")
                _model_manager.cleanup()

    except Exception as e:
        print(f"[BATCH] Error during processing: {e}")
        # 出错时强制清理
        _model_manager.cleanup()
        raise e
    
    total_time = time.time() - batch_start
    avg_time = total_time / len(batch_data)
    
    print(f"\n[BATCH] ===== BATCH SUMMARY =====")
    print(f"[BATCH] Total images: {len(batch_data)}")
    print(f"[BATCH] Total time: {total_time:.2f}s")
    print(f"[BATCH] Average per image: {avg_time:.3f}s")
    print(f"[BATCH] GPU inference time: {inference_time:.2f}s")
    print(f"[BATCH] Speedup: {(len(batch_data) * 1.441) / total_time:.2f}x")  # 基于原始1441ms/帧
    print(f"[BATCH] =======================\n")
    
    return True

def force_cleanup():
    """强制清理所有资源的外部接口"""
    global _model_manager
    _model_manager.cleanup()