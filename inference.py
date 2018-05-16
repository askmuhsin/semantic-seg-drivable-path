import os.path
import helper
import warnings
import tensorflow as tf
import project_tests as tests
from distutils.version import LooseVersion

import numpy as np
from moviepy.editor import VideoFileClip


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, l3, l4, l7

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_l7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    upsample_l7 = tf.layers.conv2d_transpose(conv_l7, num_classes, 4, 2,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    conv_l4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    fuse_l7_l4 = tf.add(upsample_l7, conv_l4)

    upsample_l4 = tf.layers.conv2d_transpose(fuse_l7_l4, num_classes, 4, 2,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    conv_l3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    fuse_l4_l3 = tf.add(upsample_l4, conv_l3)

    upsample_l3 = tf.layers.conv2d_transpose(fuse_l4_l3, num_classes, 16, 8,
                    padding='same',
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return upsample_l3


def infer():
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('./saved_model/tf_model_saved_1.meta')
      new_saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))

if __name__ == '__main__':
    infer()
