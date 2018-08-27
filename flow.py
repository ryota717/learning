from __future__ import print_function

from numba import jit
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
# ----------------yolo_module/sort.py---------------------------------------------------------
# ----------------yolo_module/sort.py---------------------------------------------------------


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)



class OpticalFlowTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        self.bbox = bbox[:4]
        self.time_since_update = 0
        self.id = OpticalFlowTracker.count
        OpticalFlowTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox[:4]

    def hokan(self,bbox):
        self.bbox == bbox[:4]
        self.hit_streak += 1


    def predict(self, good_new, good_old, mask):
        """
        good_old：オプティカルフローの始点集合(list)
        good_new：オプティカルフローの終点集合(list)
        インデックスはp0s,p1sで共通
        """

        move = 0
        vector = [0,0,0,0]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if self.bbox[0] <= c <= self.bbox[2] and self.bbox[1] <= d <= self.bbox[3] and mask[int(d), int(c)]==1:
                if (a-c) ** 2 + (b-d) ** 2 > move:
                    vector = (a-c, b-d)
        for i in range(4):
            self.bbox[i] += vector[i % 2]

        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.bbox)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.bbox


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=100, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(  winSize  = (15,15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def update(self, dets, old_gray, new_gray, mask):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1

        # calculate optical flow
        if self.trackers:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **self.feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **self.lk_params)

            good_new = p1[st==1]
            good_old = p0[st==1]

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(good_new, good_old, mask)
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)

        if dets == []:
            #update mask
            mask *= 0
            for i in self.trackers:
                box = i.get_state()
                mask[box[1]:box[3], box[0]:box[2]] += 1
            i = len(self.trackers)
            for trk in reversed(self.trackers):
                d = trk.get_state()
                if(trk.time_since_update <= self.max_age):
                    # +1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
                i -= 1
                # remove dead tracklet
                if(trk.time_since_update > self.max_age):
                    self.trackers.pop(i)
            if(len(ret) > 0):
                return mask, np.concatenate(ret)
            return mask, []

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
            else:
                trk.hokan(trk.bbox)

        #update mask
        mask *= 0
        for i in self.trackers:
            box = i.get_state()
            mask[box[1]:box[3], box[0]:box[2]] += 1

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = OpticalFlowTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return mask, np.concatenate(ret)
        return mask, np.empty((0, 5))
