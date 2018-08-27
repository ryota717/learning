from __future__ import print_function

from numba import jit
from flow import *
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
import cv2

video_path = input("input the video name : ")
vid = cv2.VideoCapture(video_path)

frame_h, frame_w = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

mot_tracker = Sort()

mask = np.ones((frame_h,frame_w),dtype=np.uint8)
old_gray = np.zeros_like(mask)
dets = np.array([[831,156,1002,249,100]], dtype = np.uint16)
num = 0

while True:
    print(num)
    ret, frame = vid.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    mask, track_bbs_ids = mot_tracker.update(dets, old_gray, frame_gray, mask)
    print(track_bbs_ids)
    print(mask[mask == 1])
    old_gray = frame_gray

    dets = []

    num += 1
