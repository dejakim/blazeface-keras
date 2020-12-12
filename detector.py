'''
detector.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python detector.py <input file>
'''

from __future__ import print_function

import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

import tensorflow as tf

from utils import jaccard_overlap, decode, anchor2rect, to_point
from blazeface import def_boxes, loss


class Detector:
  def __init__(self, thres=.95, nms_thres=.3):
    # load saved model
    model = tf.keras.models.load_model(
      './model/blazeface.h5',
      custom_objects={'loss': loss})

    W,H,C = model.input.get_shape()[1:4] # None, width, height, channel
    print('model loaded: input shape=', (W,H,C))
    self._model = model
    self._W = W
    self._C = C
    self._def_boxes = def_boxes
    self._thres = thres
    self._nms_thres = nms_thres
  
  def __nms(self, boxes, probs):
    n = len(boxes)
    if n < 1:
      return []
    # Sort descending
    idx = np.argsort(probs)[::-1]
    keep = [True] * n
    res = []
    for i in range(n-1):
      p = idx[i]
      if not keep[p]:
        continue
      agg = np.array(boxes[p]) * probs[p]
      wgt = probs[p]
      for j in range(i+1,n):
        q = idx[j]
        iou = jaccard_overlap(boxes[p], boxes[q])
        if iou > self._nms_thres:
          keep[q] = False
          agg += np.array(boxes[q]) * probs[q]
          wgt += probs[q]
      res.append( agg / wgt )
    return res
  
  def inference(self, img):
    W,C = self._W, self._C
    # Resize image
    h,w = img.shape[:2]
    src = cv2.resize(img, (W,W)).reshape(-1,W,W,C)
    # Inference
    infer = self._model.predict(src)[0]
    # Decode box
    boxes, probs = [], []
    for ib, db in zip(infer, self._def_boxes):
      prob = ib[4]
      if prob > self._thres:
        # print(prob)
        boxes.append( anchor2rect(decode(ib[:4], db), sx=w, sy=h) )
        probs.append( prob )
    # Post-processing
    boxes = self.__nms(boxes, probs)
    return boxes


if __name__ == '__main__':
  # Check input parameters
  if len(sys.argv) != 2:
    exit('usage: python {} <input file>'.format(sys.argv[0]))
  
  src_path = sys.argv[1]

  # Load source image
  img = cv2.imread(src_path, cv2.IMREAD_COLOR)
  if img is None:
    exit('could not open file: {}'.format(src_path))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  print(img.shape)

  # crop
  h,w = img.shape[:2]
  ow = min(h,w)
  ox,oy = int((w-ow)/2), int((h-ow)/2)
  img = img[oy:oy+ow, ox:ox+ow]

  # Load model
  detector = Detector()

  # Inference
  boxes = detector.inference(img)
  print("boxes = ", len(boxes))
  
  cmap = [(0,0,0), (0,255,0), (255,0,0)]
  for box in boxes:
    img = cv2.rectangle(img, to_point(box[:2]), to_point(box[2:]), cmap[1], 1)

  plt.imshow(img)
  plt.show()
