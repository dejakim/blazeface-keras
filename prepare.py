'''
prepare.py
Author: Daewung Kim (skywalker.deja@gmail.com)

Usage: python prepare.py
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import pickle
import gzip
import random
from tqdm import tqdm

from utils import anchor2rect, ground_truth, debug, draw
from blazeface import W, def_boxes, def_boxes_rect


def load_webfaces(path):
  ''' Parse Caltech Web Faces dataset

  Args:
    path (str) : A path where the Caltech Web Fases dataset decompressed
  
  Returns:
    list : A list of image path (str), ground truth boxes (2-D list) pair
           each box can be parsed as left, top, right, bottom (0 ~ 1.0)
  
  '''
  print("loading caltech web faces data set...")
  # check if path is available
  if not os.path.isdir(path):
    print("dataset is not exists... skip: " + path)
    return
  # load metadata
  txt_file = os.path.join(path, "WebFaces_GroundThruth.txt")
  if not os.path.isfile(txt_file):
    exit('Could not open metadata file.')
  
  f = open(txt_file, 'r')
  dic = {}
  lines = f.readlines()
  for line in lines:
    tok = line.split()
    # each line contains 9 values
    if len(tok) != 9:
      continue
    img_path, ann = tok[0], tok[1:]
    lx, ly, rx, ry, nx, ny, mx, my = list(map(lambda x: float(x), ann))
    if img_path not in dic:
      dic[img_path] = []
    # Determining face area :
    # since original ground truth includes eyes, nose and mouth points only,
    # I added a margin to the rect area, which includes the eyes, nose and mouth.
    l, r = min(lx,rx,nx,mx), max(lx,rx,nx,mx)
    t, b = min(ly,ry,ny,my), max(ly,ry,ny,my)
    w, h = r-l, b-t
    l, r, t, b = l-w/2, r+w/2, t-h/2, b+h/2
    dic[img_path].append([l, t, r, b])
  f.close()
  # serialize
  data = []
  for img_path in dic:
    data.append([os.path.join(path, "img", img_path) , dic[img_path]])
  return data

if __name__ == '__main__':
  bin_path = './data/faces_raw.pickle'
  enc_path = './data/faces_encoded.pickle'

  print('-'*30)
  print('load meta data')
  data = load_webfaces('./data/Caltech_WebFaces')

  if not data:
    exit('no data for pre-processing')

  print('-'*30)
  print('load image and annotation')
  x_data, y_data, y_data_enc = [], [], []
  for i in tqdm(range(len(data))):
    img_path, obj_boxes = data[i]
    # image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
      print("could not open file: " + img_path); continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize and crop
    h,w = img.shape[:2]
    ow = min(h,w)
    ox,oy = int((w-ow) * .5), int((h-ow) * .5)

    # boxes
    y = []
    for l,t,r,b in obj_boxes:
      bx,by = (r+l) * .5, (b+t) * .5
      bw = max(r-l, b-t)
      if bx > ox and bx < ox+ow and by > oy and by < oy+ow:
        box = np.array([bx-ox, by-oy, bw, bw], dtype=np.float32) / ow
        y.append(box)
    if not y:
      continue
    y_enc = ground_truth(y, def_boxes, def_boxes_rect)
    
    # resize and crop original image
    x = cv2.resize(img[oy:oy+ow, ox:ox+ow], (W,W))

    # if i < 30:
    #   debug(x, y)
    #   draw(x, y_enc, def_boxes)
    # else:
    #   exit("Quit")

    # add image and ground truth
    x_data.append(x)
    y_data.append(y)
    y_data_enc.append(y_enc)
  
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  y_data_enc = np.array(y_data_enc, dtype=np.float32)
  print(x_data.shape, y_data.shape, y_data_enc.shape)

  print('-'*30)
  print('save to file')
  # save raw blob
  with gzip.open(bin_path, 'wb') as f:
    pickle.dump({ "x_data":x_data, "y_data":y_data }, f)
  # save encode blob
  with gzip.open(enc_path, 'wb') as f:
    pickle.dump({ "x_data":x_data, "y_data":y_data_enc }, f)
  
  print('done')