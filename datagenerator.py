'''
datagenerator.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np
import cv2
import pickle
import gzip

import tensorflow as tf
from utils import anchor2rect, ground_truth, draw
from blazeface import def_boxes, def_boxes_rect

pi_4 = np.pi * .25

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, images, labels, batch_size=16, shuffle=True, random_state=33):
    '''Initialization'''
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    self.shuffle = shuffle
    # custom params
    self.random_state = random_state
    self.image_size = images[0].shape[:2][::-1] # row,col -> col,row
    self.pivot = (self.image_size[0] * .5, self.image_size[1] * .5)
    # initial indexes
    self.on_epoch_end()

  def __len__(self):
    return len(self.images) // self.batch_size

  def __getitem__(self, index):
    # Generate indexes of the batch
    start = index * self.batch_size
    indexes = self.indexes[ start : start + self.batch_size ]

    # Generate data
    return self.__data_generation(indexes)

  def on_epoch_end(self):
    '''Updates indexes after each epoch'''
    self.indexes = np.arange(len(self.images))
    if self.shuffle == True:
      np.random.seed(self.random_state)
      np.random.shuffle(self.indexes)

  def __data_generation(self, indexes):
    angles = np.random.rand(len(indexes))
    x, y = [], []
    for i, a in zip(indexes, angles):
      image, label = self.images[i], self.labels[i]
      # get rotation matrix
      r, d = a * pi_4, a * 45. # radian, degree
      s = 1. / (np.cos(r) + np.sin(r))
      R = cv2.getRotationMatrix2D(self.pivot, d, s)
      # rotate & scale image
      x.append( cv2.warpAffine(image, R, self.image_size, borderMode=cv2.BORDER_REPLICATE) )
      # rotate & scale ground truth boxes
      R = cv2.getRotationMatrix2D((.5,.5), d, s)
      boxes = []
      for box in label:
        cx,cy = np.matmul(R, [box[0], box[1], 1.])
        boxes.append([cx, cy, box[2] * s, box[3] * s])
      y.append( ground_truth(boxes, def_boxes, def_boxes_rect) )
    return np.array(x), np.array(y, dtype=np.float32)

if __name__ == "__main__":
  bin_path = './data/faces_raw.pickle'

  if not os.path.isfile(bin_path):
    exit("Could not load data at: " + bin_path)
  
  print("load files")
  with gzip.open(bin_path, 'rb') as f:
    data = pickle.load(f)
    x_data, y_data = data["x_data"], data["y_data"]
    # shuffle
    idx = np.arange(len(x_data)); np.random.shuffle(idx)
    x_train, y_train = x_data[idx], y_data[idx]
  
  print("generate image data")
  datagen = DataGenerator(x_train, y_train)
  for i, (x,y) in enumerate(datagen):
    if i > 0:
      break
    for x_i, y_i in zip(x,y):
      draw(x_i, y_i, def_boxes)