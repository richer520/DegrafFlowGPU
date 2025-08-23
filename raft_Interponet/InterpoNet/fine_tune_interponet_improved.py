# fine_tune_interponet_improved.py

import os
import numpy as np
import tensorflow as tf
import io_utils
import utils
import model
from glob import glob
import skimage.io

# ========== 配置 ==========
downscale = 8
epochs = 200  # 增加epoch数
batch_size = 3  # 减小batch size以适应更复杂的损失
checkpoint_dir = "./checkpoints_improved"
os.makedirs(checkpoint_dir, exist_ok=True)

# 验证集分割比例
validation_split = 0.2

# ========== 加载数据路径 ==========
data_dir = "./training_data"
image_dir = os.path.join(data_dir, "images")
edges_dir = os.path.join(data_dir, "edges")
matches_dir = os.path.join(data_dir, "matches")
gt_flo_dir = os.path.join(data_dir, "gt_flo")

# ========== 横向依赖损失函数 ==========
def lateral_dependency_loss(pred, gt):
    """
    实现InterpoNet论文中的横向依赖损失
    """
    # 计算相邻像素的光流距离
    pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    gt_dx = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    gt_dy = gt[:, 1:, :, :] - gt[:, :-1, :, :]
    
    # 计算EPE距离
    pred_dist_x = tf.sqrt(tf.reduce_sum(tf.square(pred_dx), axis=3))
    pred_dist_y = tf.sqrt(tf.reduce_sum(tf.square(pred_dy), axis=3))
    gt_dist_x = tf.sqrt(tf.reduce_sum(tf.square(gt_dx), axis=3))
    gt_dist_y = tf.sqrt(tf.reduce_sum(tf.square(gt_dy), axis=3))
    
    # 横向依赖损失
    lateral_loss_x = tf.reduce_mean(tf.abs(pred_dist_x - gt_dist_x))
    lateral_loss_y = tf.reduce_mean(tf.abs(pred_dist_y - gt_dist_y))
    
    return lateral_loss_x + lateral_loss_y

# ========== 数据增强函数 ==========
def augment_data(sparse_img, mask, edges, gt_flow):
    """
    模拟RAFT采样的噪声特征
    """
    # 30%概率添加噪声
    if np.random.random() < 0.3:
        # 随机选择5%的有效像素添加噪声
        valid_pixels = mask > 0
        noise_ratio = 0.05
        num_noise_pixels = int(np.sum(valid_pixels) * noise_ratio)
        
        if num_noise_pixels > 0:
            valid_indices = np.where(valid_pixels)
            noise_indices = np.random.choice(len(valid_indices[0]), num_noise_pixels, replace=False)
            
            # 添加高斯噪声，模拟RAFT采样误差
            noise_y = valid_indices[0][noise_indices]
            noise_x = valid_indices[1][noise_indices]
            sparse_img[noise_y, noise_x] += np.random.normal(0, 1.5, (num_noise_pixels, 2))
    
    # 20%概率进行水平翻转
    if np.random.random() < 0.2:
        sparse_img = np.fliplr(sparse_img)
        sparse_img[:, :, 0] = -sparse_img[:, :, 0]  # 翻转x方向光流
        mask = np.fliplr(mask)
        edges = np.fliplr(edges)
        gt_flow = np.fliplr(gt_flow)
        gt_flow[:, :, 0] = -gt_flow[:, :, 0]  # 翻转x方向光流
    
    return sparse_img, mask, edges, gt_flow

# ========== 构建模型 ==========
graph = tf.Graph()
with graph.as_default():
    # 占位符
    image_ph = tf.placeholder(tf.float32, shape=(None, None, None, 2))
    mask_ph = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    edges_ph = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    gt_flow_ph = tf.placeholder(tf.float32, shape=(None, None, None, 2))
    is_training_ph = tf.placeholder(tf.bool, shape=())

    # 网络预测
    prediction = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)

    # EPE损失
    epe_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(prediction - gt_flow_ph), axis=3)))
    
    # 横向依赖损失
    ld_loss = lateral_dependency_loss(prediction, gt_flow_ph)
    
    # 总损失（可调权重）
    total_loss = epe_loss + 0.3 * ld_loss
    
    # 学习率衰减
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=1e-5,
        global_step=global_step,
        decay_steps=50,  # 每50步衰减
        decay_rate=0.95,
        staircase=True
    )
    
    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    
    # 保存器
    saver = tf.train.Saver(max_to_keep=5)

# ========== 数据准备 ==========
def load_data_list():
    image_files = sorted(glob(os.path.join(image_dir, "*_10.png")))
    
    # 随机打乱并分割训练集和验证集
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - validation_split))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    return train_files, val_files

def load_sample(img10_path, apply_augmentation=False):
    """加载单个训练样本"""
    # 动态获取图像尺寸
    img1 = skimage.io.imread(img10_path)
    img_height, img_width = img1.shape[:2]

    idx = os.path.basename(img10_path).replace("_10.png", "")
    edge_path = os.path.join(edges_dir, f"{idx}_edges.dat")
    match_path = os.path.join(matches_dir, f"{idx}_matches.txt")
    gt_path = os.path.join(gt_flo_dir, f"{idx}_10.flo")

    # 加载数据
    edges = io_utils.load_edges_file(edge_path, img_width, img_height)
    sparse_img, mask = io_utils.load_matching_file(match_path, img_width, img_height)
    gt_flow = io_utils.load_flow_file(gt_path)

    # 数据增强
    if apply_augmentation:
        sparse_img, mask, edges, gt_flow = augment_data(sparse_img, mask, edges, gt_flow)

    # 下采样
    sparse_img, mask, edges = utils.downscale_all(sparse_img, mask, edges, downscale)
    gt_flow_dummy_mask = np.ones((gt_flow.shape[0], gt_flow.shape[1]))
    gt_flow, _, _ = utils.downscale_all(gt_flow, gt_flow_dummy_mask, None, downscale)

    return sparse_img, mask, edges, gt_flow

def evaluate_validation(sess, val_files):
    """评估验证集"""
    total_val_loss = 0
    total_epe_loss = 0
    
    for val_file in val_files:
        sparse_img, mask, edges, gt_flow = load_sample(val_file, apply_augmentation=False)
        
        feed_dict = {
            image_ph: np.expand_dims(sparse_img, axis=0),
            mask_ph: np.expand_dims(mask[..., np.newaxis], axis=0),
            edges_ph: np.expand_dims(edges[..., np.newaxis], axis=0),
            gt_flow_ph: np.expand_dims(gt_flow, axis=0),
            is_training_ph: False
        }

        val_total_loss, val_epe_loss = sess.run([total_loss, epe_loss], feed_dict=feed_dict)
        total_val_loss += val_total_loss
        total_epe_loss += val_epe_loss

    return total_val_loss / len(val_files), total_epe_loss / len(val_files)

# ========== 训练流程 ==========
def train():
    train_files, val_files = load_data_list()
    
    # 早停机制
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        
        # 优先加载KITTI 2012预训练模型
        kitti2012_model_path = "./models/df_kitti2012.ckpt"
        loaded_pretrained = False
        
        if os.path.exists(kitti2012_model_path + ".index"):
            try:
                saver.restore(sess, kitti2012_model_path)
                print(f"Successfully loaded KITTI 2012 pretrained model: {kitti2012_model_path}")
                loaded_pretrained = True
            except Exception as e:
                print(f"Failed to load KITTI 2012 model: {e}")
        
        # 如果预训练模型加载失败，尝试加载之前的检查点
        if not loaded_pretrained:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                saver.restore(sess, latest_checkpoint)
                print(f"Restored from checkpoint: {latest_checkpoint}")
            else:
                print("Training from scratch (no pretrained model found)")
        
        for epoch in range(epochs):
            # 训练阶段
            np.random.shuffle(train_files)  # 每个epoch重新打乱
            total_train_loss = 0
            total_train_epe = 0
            total_train_ld = 0
            
            for i, img_file in enumerate(train_files):
                sparse_img, mask, edges, gt_flow = load_sample(img_file, apply_augmentation=True)
                
                feed_dict = {
                    image_ph: np.expand_dims(sparse_img, axis=0),
                    mask_ph: np.expand_dims(mask[..., np.newaxis], axis=0),
                    edges_ph: np.expand_dims(edges[..., np.newaxis], axis=0),
                    gt_flow_ph: np.expand_dims(gt_flow, axis=0),
                    is_training_ph: True
                }

                _, batch_total_loss, batch_epe_loss, batch_ld_loss, current_lr = sess.run(
                    [train_op, total_loss, epe_loss, ld_loss, learning_rate], 
                    feed_dict=feed_dict
                )
                
                total_train_loss += batch_total_loss
                total_train_epe += batch_epe_loss
                total_train_ld += batch_ld_loss
                
                # 每20个样本打印一次进度
                if (i + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}, Sample {i+1}/{len(train_files)}, "
                          f"Loss: {batch_total_loss:.6f}, EPE: {batch_epe_loss:.6f}, "
                          f"LD: {batch_ld_loss:.6f}, LR: {current_lr:.2e}")

            # 验证阶段
            val_total_loss, val_epe_loss = evaluate_validation(sess, val_files)
            
            avg_train_loss = total_train_loss / len(train_files)
            avg_train_epe = total_train_epe / len(train_files)
            avg_train_ld = total_train_ld / len(train_files)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train - Total: {avg_train_loss:.6f}, EPE: {avg_train_epe:.6f}, LD: {avg_train_ld:.6f}")
            print(f"  Val   - Total: {val_total_loss:.6f}, EPE: {val_epe_loss:.6f}")
            
            # 早停检查
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                patience_counter = 0
                # 保存最佳模型
                save_path = saver.save(sess, os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.ckpt"))
                print(f"  New best model saved: {save_path}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epochs")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.ckpt"))
                print(f"  Regular checkpoint saved: {save_path}")
            
            print("-" * 80)

if __name__ == "__main__":
    print("Starting improved InterpoNet fine-tuning...")
    print("Key improvements:")
    print("- EPE + Lateral Dependency Loss")
    print("- Learning rate decay")
    print("- Data augmentation for RAFT noise")
    print("- Validation set and early stopping")
    print("- Comprehensive logging")
    print("-" * 80)
    
    train()