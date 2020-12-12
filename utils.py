'''
utils.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt

def to_point(pt):
  return (int(pt[0]), int(pt[1]))

def anchor2rect(anchor, sx=1., sy=1.):
  x,y,w,h = anchor
  x,y,w,h = x * sx, y * sy, w * sx * .5, h * sy * .5
  return [x-w, y-h, x+w, y+h]

def rect2anchor(rect, sx=1., sy=1.):
  l,t,r,b = rect
  l,t,r,b = l * sx, t * sy, r * sx, b * sy
  return [(l+r) * .5, (t+b) * .5, r-l, b-t]

def decode(infer, bbox):
  cx, cy, dw, dh = bbox
  gx, gy, gw, gh = infer
  x, y = (gx * dw + cx), (gy * dh + cy)
  w, h = np.exp(gw) * dw, np.exp(gh) * dh
  return [x, y, w, h]

def encode(gt, bbox):
  cx, cy, dw, dh = bbox
  gx, gy, gw, gh = gt
  x, y = (gx - cx) / dw, (gy - cy) / dh
  w, h = np.log(gw / dw), np.log(gh / dh)
  return [x, y, w, h, 1.]

def jaccard_overlap(src, dst):
  l0,t0,r0,b0 = src
  l1,t1,r1,b1 = dst
  intersect = max(min(r0,r1) - max(l0,l1), 0) * max(min(b0,b1) - max(t0,t1), 0)
  return intersect / ((r0-l0) * (b0-t0) + (r1-l1) * (b1-t1) - intersect)

def default_boxes(mk, wk):
  boxes = []
  for m, w in zip(mk, wk):
    grid = np.arange(m)/m + .5/m
    for cy in grid:
      for cx in grid:
        for w_ in w:
          boxes.append( (cx, cy, w_, w_) )
  return boxes

def ground_truth(boxes, priors, priors_rect=None):
  scores, y = [], []
  gts = [ anchor2rect(box) for box in boxes ]
  # iou score
  if not priors_rect:
    for prior in priors:
      dst = anchor2rect(prior)
      iou = [ jaccard_overlap(gt, dst) for gt in gts ]
      scores.append(iou)
  else:
    for prior in priors_rect:
      iou = [ jaccard_overlap(gt, prior) for gt in gts ]
      scores.append(iou)
  # To include at least one box,
  # add 1.0 to maximum iou prior
  scores = np.array(scores)
  for i in range(len(boxes)):
    idx = np.argmax(scores[:,i])
    scores[idx,i] += 1.
  # encode ground truth boxes
  for iou, prior in zip(scores, priors):
    idx = np.argmax(iou)
    if iou[idx] < .5:
      y.append( [0.] * 5 )
    else:
      y.append( encode(boxes[idx], prior) )
  return y

def debug(img, priors):
  h,w = img.shape[:2]
  for bb in priors:
    box = anchor2rect( bb, sx=w, sy=h )
    img = cv2.rectangle(img, to_point(box[:2]), to_point(box[2:]), (0,255,0), 1)
  plt.imshow(img)
  plt.show()

def draw(img, infers, priors):
  h,w = img.shape[:2]
  for ib, bb in zip(infers, priors):
    if ib[4] > 0:
      # expected box
      box = anchor2rect( decode(ib[:4], bb), sx=w, sy=h )
      img = cv2.rectangle(img, to_point(box[:2]), to_point(box[2:]), (0,255,0), 1)
      # matched default box
      dbx = anchor2rect( bb, sx=w, sy=h )
      img = cv2.rectangle(img, to_point(dbx[:2]), to_point(dbx[2:]), (255,0,0), 1)
  plt.imshow(img)
  plt.show()