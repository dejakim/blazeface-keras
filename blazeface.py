'''
blazeface.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np
import cv2
import pickle
import gzip

import tensorflow as tf
from utils import anchor2rect, default_boxes

# Tensorflow dimension ordering
tf.keras.backend.set_image_data_format('channels_last')

eps = 1e-7

# input width
W = 128
# number of priors for anchor computation
num_priors = [2, 6]

# map size
mk = [16, 8]
# scale
sk = np.linspace(0.08, 0.45, sum(num_priors))
# scaled width per map
wk = [sk[:2], sk[2:]]
# defalut boxes (a.k.a priors)
def_boxes = default_boxes(mk, wk)
def_boxes_rect = [ anchor2rect(box) for box in def_boxes ]


def binary_crossentropy(y_true, y_pred):
  y = tf.clip_by_value(y_pred, eps, 1. - eps)
  return tf.where(tf.equal(y_true,1.), -1.* tf.math.log(y), -1.* tf.math.log(1.-y))

def squared_error(y_true, y_pred):
  return tf.math.squared_difference(y_true, y_pred)

def smooth_l1_loss(y_true, y_pred):
  abs_loss = tf.abs(tf.subtract(y_true, y_pred))
  sqr_loss = .5 * tf.math.squared_difference(y_true, y_pred)
  loss = tf.where(tf.less(abs_loss, 1.), sqr_loss, abs_loss - .5)
  return tf.reduce_sum(loss, axis=-1)

def loss(y_true, y_pred):
  loc_true = tf.reshape(y_true[:,:,:4], [-1, 4])
  loc_pred = tf.reshape(y_pred[:,:,:4], [-1, 4])
  conf_true = tf.reshape(y_true[:,:,4:], [-1])
  conf_pred = tf.reshape(y_pred[:,:,4:], [-1])

  # compute mask
  pos_mask = tf.equal(conf_true, 1.)
  pos_mask_float = tf.cast(pos_mask, tf.float32)
  num_pos = tf.reduce_sum(pos_mask_float)
  neg_mask = tf.logical_not(pos_mask)
  neg_mask_float = tf.cast(neg_mask, tf.float32)
  num_neg = tf.reduce_sum(neg_mask_float)

  # confidence loss
  ce = binary_crossentropy(conf_true, conf_pred)
  # ce = squared_error(conf_true, conf_pred)
  positives = tf.reduce_sum(tf.where(pos_mask, ce, 0.))
  negatives = tf.reduce_sum(tf.where(neg_mask, ce, 0.))
  conf_loss = positives / num_pos + negatives / num_neg

  # Online Hard Example Mining
  # # num_neg_top = tf.cast(tf.minimum(num_pos * 3.0, num_neg), tf.int32)
  # num_neg_top = tf.cast(tf.maximum(num_neg / 3., num_pos * 3.), tf.int32)
  # negatives = tf.math.top_k(tf.where(neg_mask, ce, 0.), k=num_neg_top)[0]
  # negatives = tf.reduce_sum(negatives)
  # conf_loss = positives/num_pos + negatives/tf.cast(num_neg_top, tf.float32)

  # location loss
  loc_loss = smooth_l1_loss(loc_true, loc_pred)
  loc_loss = tf.reduce_sum(tf.where(pos_mask, loc_loss, 0.)) / num_pos

  return conf_loss + loc_loss


def single_block(x, filters=24, strides=1, padding='same'):
  # convolution path
  y = tf.keras.layers.SeparableConv2D(
    filters,
    kernel_size=5,
    strides=strides,
    padding=padding,
    use_bias=False)(x)
  y = tf.keras.layers.BatchNormalization()(y)

  # residual path
  if strides == 2:
    x = tf.keras.layers.MaxPooling2D()(x)
    _,w,h,c = x.shape[:4] # batch, width, height, channel
    if c < filters:
      pad = tf.zeros_like(x)
      x = tf.keras.layers.concatenate([x, pad], axis=-1)
  
  z = tf.keras.layers.Add()([y, x])
  z = tf.keras.layers.Activation("relu")(z)
  return z

def double_block(x, filters0=24, filters=96, strides=1, padding='same', dr=0.3):
  # convolution projection
  y = tf.keras.layers.SeparableConv2D(
    filters=filters0,
    kernel_size=5,
    strides=strides,
    padding=padding,
    use_bias=False)(x)
  y = tf.keras.layers.BatchNormalization()(y)
  y = tf.keras.layers.Activation("relu")(y)

  # convolution expand
  y = tf.keras.layers.SeparableConv2D(
    filters,
    kernel_size=5,
    strides=1,
    padding=padding,
    use_bias=False)(y)
  y = tf.keras.layers.BatchNormalization()(y)
  y = tf.keras.layers.Activation("relu")(y)

  # residual path
  if strides == 2:
    x = tf.keras.layers.MaxPooling2D()(x)
    _,w,h,c = x.shape[:4] # batch, width, height, channel
    if c < filters:
      pad = tf.zeros_like(x)
      x = tf.keras.layers.concatenate([x, pad], axis=-1)

  z = tf.keras.layers.Add()([y, x])
  z = tf.keras.layers.Activation("relu")(z)
  z = tf.keras.layers.Dropout(dr)(z)
  return z

def conv_block(x, filters=24, strides=1, padding='same'):
  x = tf.keras.layers.Conv2D(filters, 5, strides=strides, padding=padding)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation("relu")(x)
  return x

def classifier(layers, priors):
  loc, conf = [], []
  for x, p in zip(layers, priors):
    # location
    x1 = tf.keras.layers.Conv2D(p * 4, 3, padding='same')(x)
    x1 = tf.keras.layers.Reshape((-1, 4))(x1)
    loc.append(x1)
    # confidence
    x2 = tf.keras.layers.Conv2D(p, 5, padding='same')(x)
    x2 = tf.keras.layers.Reshape((-1, 1))(x2)
    x2 = tf.keras.layers.Activation('sigmoid')(x2)
    conf.append(x2)
  # concatenate
  loc = tf.keras.layers.concatenate(loc, axis=1)
  conf = tf.keras.layers.concatenate(conf, axis=1)
  output = tf.keras.layers.concatenate([loc, conf], axis=-1)
  
  return output


def create_model(input_shape=(W,W,3)):
  inputs = tf.keras.layers.Input(shape=input_shape)
  # convolution
  x = conv_block(inputs, 24, 2) # 128x128 -> 64x64
  # single block
  x = single_block(x, filters=24, strides=1) # 64x64
  x = single_block(x, filters=24, strides=1) # 64x64
  x = single_block(x, filters=48, strides=2) # 64x64 -> 32x32
  x = single_block(x, filters=48, strides=1) # 32x32
  x = single_block(x, filters=48, strides=1) # 32x32
  # double block
  y = double_block(x, strides=2) # 32x32 -> 16x16
  y = double_block(y, strides=1) # 16x16
  y = double_block(y, strides=1) # 16x16
  z = double_block(y, strides=2) # 16x16 -> 8x8
  z = double_block(z, strides=1) # 8x8
  z = double_block(z, strides=1) # 8x8
  # classifier
  outputs = classifier([y,z], num_priors)

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  # opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, decay=1e-4)
  opt = tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True)
  model.compile(optimizer=opt, loss=loss)#, metrics=['accuracy'])

  return model