'''
app.py
Author: Daewung Kim (skywalker.deja@gmail.com)
'''
from __future__ import print_function

import os
import numpy as np
import cv2

from utils import to_point
from detector import Detector

def crop(img):
  h,w = img.shape[:2]
  ow = min(h,w)
  ox,oy = int((w-ow)/2), int((h-ow)/2)
  return img[oy:oy+ow, ox:ox+ow]

if __name__ == '__main__':
  # Load model
  detector = Detector()
  # camera open
  capture = cv2.VideoCapture(0)
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  # main loop
  while True:
    ret, frame = capture.read()
    # crop
    img = crop(frame)
    # Inference
    boxes = detector.inference(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for box in boxes:
      img = cv2.rectangle(img, to_point(box[:2]), to_point(box[2:]), (0,255,0), 1)
    
    cv2.imshow("VideoFrame", img)
    if cv2.waitKey(1) > 0: break

  capture.release()
  cv2.destroyAllWindows()
