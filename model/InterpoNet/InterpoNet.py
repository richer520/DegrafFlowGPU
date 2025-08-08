import numpy as np
import tensorflow as tf
import skimage as sk
import utils
import io_utils
import model
import argparse

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

            utils.calc_variational_inference_map(img1_filename, img2_filename,
                                                 out_filename, out_filename,
                                                 dataset_type)

            print(f"Saved flow to: {out_filename}")
