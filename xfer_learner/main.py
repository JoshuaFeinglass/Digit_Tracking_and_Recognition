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
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (10,10),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

#ESTABLISH VIDEO STREAM
frame_processed = 0
score_thresh = 0.2
num_hands_detect = 1

# Read video
video = cv2.VideoCapture(-1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
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

k = cv2.waitKey(1) & 0xff
detection_graph, sess = detector_utils.load_inference_graph()
detected = 0
color = np.random.randint(0,255,(100,3))
num_detects = deque(maxlen=30)
q = deque(maxlen=120)
#FRAME LOOP
while True:
    timer = cv2.getTickCount()
    if detected:
        old_frame = frame
    ok, frame = video.read()
    frame = cv2.flip(frame,2)
    if not ok:
        break
    display_frame = frame
    if not detected:
        boxes, scores = detector_utils.detect_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),detection_graph,sess)
        q.append(scores[0])
        if np.mean(q) > score_thresh:
            num_detects.append(1)
        else:
            num_detects.append(0)
        if np.sum(num_detects) > 20:
            bbox = convert_to_tracking_format(boxes,0)#np.argmax(scores))
            mask = np.zeros_like(frame)
            #drawing_mask = np.zeros_like(frame)
            path_mask = np.zeros_like(frame)
            mask[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3],:] = 255
            mask = mask[:,:,0]
            tracker = cv2.TrackerMedianFlow_create()
            ok = tracker.init(frame, bbox)
            detected = 1
            loop_num = 0
    else:
        cX = int(bbox[0]+bbox[2]/2)
        cY = int(bbox[1]+bbox[3]/2)

        ok, bbox_estimate = tracker.update(frame)
        if ok:
            bbox = bbox_estimate
        else:
            boxes, scores = detector_utils.detect_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),detection_graph,sess)
            bbox = convert_to_tracking_format(boxes,0)
            ok = tracker.init(frame, bbox)
        print(bbox)
        mask = np.zeros_like(frame)
        mask[int(bbox[0]):int(bbox[0] + bbox[2]), int(bbox[1]):int(bbox[1] + bbox[3]), :] = 255
        mask[mask>0]=255
        print(cX)
        print(cY)
        path_mask[cY - 10:cY + 10, cX - 10:cX + 10] = 255
        cv2.circle(display_frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.imshow("path",path_mask)
        loop_num += 1
        cv2.putText(display_frame, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        loop_num+=1
        if loop_num >= 180:
            detected = 0
            num_detects = deque(maxlen=30)
            q = deque(maxlen=120)

    if detected:
        # draw bounding boxes on frame
        point1 = (int(bbox[0]),int(bbox[1]))

        point2 = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
        cv2.rectangle(display_frame, point1, point2, (77, 255, 9), 3, 1)
    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 	# Calculate Frames per second (FPS)
    cv2.putText(display_frame, str(np.mean(q)) + " confidence", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                     display_frame)
    cv2.imshow("Tracking", cv2.resize(display_frame, (int(3*im_width),int(3*im_height)), interpolation = cv2.INTER_AREA))
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
	   