from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import cv2
import sys
import numpy as np
from collections import deque

def convert_to_tracking_format(detect,ind):
    (left, right, top, bottom) = (detect[ind][1] * im_width, detect[ind][3] * im_width,
                                  detect[ind][0] * im_height, detect[ind][2] * im_height)
    
    bounding_box = (int(left),int(top),int(right-left),int(bottom-top))
    return bounding_box


#ESTABLISH VIDEO STREAM
frame_processed = 0
score_thresh = 0.3
num_hands_detect = 1

# Read video
video = cv2.VideoCapture(-1)
im_width, im_height = (video.get(3), video.get(4))
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

detection_graph, sess = detector_utils.load_inference_graph()
#tracker = cv2.TrackerMedianFlow_create()
detected = 0
#FRAME LOOP
while True:
    timer = cv2.getTickCount()
    ok, frame = video.read()
    if not ok:
        break
    # try:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # except:
    #     print("Error converting to RGB")
	####################################################
    if not detected:
        q = deque(maxlen=60)
        boxes, scores = detector_utils.detect_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),detection_graph,sess)
        q.append(scores[0])
        if 1:#q > score_thresh:
        	bbox = convert_to_tracking_format(boxes,0)#np.argmax(scores))
        	#detected = 1
        	#ok = tracker.init(frame, bbox)
	####################################################
    else:
	    # Update tracker
        ok, bbox = tracker.update(frame)
        # if tracking fails, try detecting again
        if not ok:
            boxes, scores = detector_utils.detect_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                                   detection_graph, sess)
            bbox = convert_to_tracking_format(boxes,np.argmax(scores))

    if 1: #detected:
        # draw bounding boxes on frame
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 	# Calculate Frames per second (FPS)
    cv2.putText(frame, str(np.mean(q)) + " confidence", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                     frame)
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
	   