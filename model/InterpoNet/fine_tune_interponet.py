# fine_tune_interponet.py

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
epochs = 30  # 如果你加 early stopping 可以设置大些
batch_size = 1  # InterpoNet是小模型，支持1张张训
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 随机种子设置，保证可重复性
np.random.seed(42)
tf.set_random_seed(42)

# ========== 加载数据路径 ==========
data_dir = "./training_data"
image_dir = os.path.join(data_dir, "images")
edges_dir = os.path.join(data_dir, "edges")
matches_dir = os.path.join(data_dir, "matches")
gt_flo_dir = os.path.join(data_dir, "gt_flo")

# ========== 构建模型 ==========
graph = tf.Graph()
with graph.as_default():

    image_ph = tf.placeholder(tf.float32, shape=(None, None, None, 2))
    mask_ph = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    edges_ph = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    gt_flow_ph = tf.placeholder(tf.float32, shape=(None, None, None, 2))


    prediction = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)

    loss = tf.reduce_mean(tf.square(prediction - gt_flow_ph))
    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

    saver = tf.train.Saver()

# ========== 训练流程 ==========
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    image_files = sorted(glob(os.path.join(image_dir, "*_10.png")))
    
    for epoch in range(epochs):
        total_loss = 0
        for img10_path in image_files:
            # 动态获取当前图像的真实尺寸
            img1 = skimage.io.imread(img10_path)
            img_height, img_width = img1.shape[:2]

            idx = os.path.basename(img10_path).replace("_10.png", "")
            img11_path = os.path.join(image_dir, f"{idx}_11.png")
            edge_path = os.path.join(edges_dir, f"{idx}_edges.dat")
            match_path = os.path.join(matches_dir, f"{idx}_matches.txt")
            gt_path = os.path.join(gt_flo_dir, f"{idx}_10.flo")

            edges = io_utils.load_edges_file(edge_path, img_width, img_height)
            sparse_img, mask = io_utils.load_matching_file(match_path, img_width, img_height)
            gt_flow = io_utils.load_flow_file(gt_path)

            sparse_img, mask, edges = utils.downscale_all(sparse_img, mask, edges, downscale)
            gt_flow_dummy_mask = np.ones((gt_flow.shape[0], gt_flow.shape[1]))  # 创建正确形状的mask
            gt_flow, _, _ = utils.downscale_all(gt_flow, gt_flow_dummy_mask, None, downscale)
            # gt_flow, _, _ = utils.downscale_all(gt_flow, np.ones_like(mask), None, downscale)

            feed_dict = {
                image_ph: np.expand_dims(sparse_img, axis=0),
                mask_ph: np.expand_dims(mask[..., np.newaxis], axis=0),
                edges_ph: np.expand_dims(edges[..., np.newaxis], axis=0),
                gt_flow_ph: np.expand_dims(gt_flow, axis=0),
            }

            _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
            total_loss += batch_loss

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(image_files):.6f}")

        # 保存模型
        saver.save(sess, os.path.join(checkpoint_dir, "fine_tuned_kitti2015.ckpt"))

print("Training is complete and weights are saved!")
