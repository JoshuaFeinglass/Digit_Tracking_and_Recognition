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
    
    bounding_box = (int(top),int(left),int(bottom-top),int(right-left))
    return bounding_box
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
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

#FRAME LOOP
while True:
    timer = cv2.getTickCount()
    if detected:
        old_frame = frame
    ok, frame = video.read()
    cv2.flip(frame,1)
    if not ok:
        break
    display_frame = frame
    # try:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # except:
    #     print("Error converting to RGB")
	####################################################
    if not detected:
        q = deque(maxlen=120)
        boxes, scores = detector_utils.detect_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),detection_graph,sess)
        q.append(scores[0])
        if np.mean(q) > score_thresh:
            num_detects.append(1)
        else:
            num_detects.append(0)
        if np.sum(num_detects) > 20:
            bbox = convert_to_tracking_format(boxes,0)#np.argmax(scores))
            mask = np.zeros_like(frame)
            drawing_mask = np.zeros_like(frame)
            path_mask = np.zeros_like(frame)
            mask[bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3],:] = 255
            mask = mask[:,:,0]

            #get centroid
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            M = cv2.moments(thresh)
            if M["m00"]:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                path_mask[cY - 10:cY + 10, cX - 10:cX + 10] = 255

            track_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(track_frame, mask=mask, **feature_params)
            detected = 1

    ####################################################
    else:
        p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask=mask, **feature_params)
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY),cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        else:
            detected = 0

        if good_new.size:
            trans_mat = cv2.estimateAffinePartial2D(np.vstack(good_old),np.vstack(np.round(good_new)))
        else:
            detected = 0
        #old_points = (point1,point2)
        if trans_mat[0] is not None:
            rows, cols = mask.shape
            cX = int(cX+trans_mat[0][0,2])
            cY = int(cY+trans_mat[0][1,2])
            path_mask[cY - 10:cY + 10, cX - 10:cX + 10] = 255
            #mask = cv2.warpAffine(mask, np.hstack((np.array([[1,0],[0,1]]),np.matrix(trans_mat[0][:,-1]).transpose())), (cols,rows))
            #mask[mask>0]=255
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            drawing_mask = cv2.line(drawing_mask, (a, b), (c, d), color[i].tolist(), 2)
            display_frame = cv2.circle(display_frame, (a, b), 5, color[i].tolist(), -1)
        #draw the centroid
        cv2.imshow("path",path_mask)

        cv2.circle(display_frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.putText(display_frame, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        img = cv2.add(display_frame, drawing_mask)
        p0 = good_new.reshape(-1, 1, 2)
        cv2.imshow("track",cv2.bitwise_and(np.repeat(mask[:,:,np.newaxis],3,axis=2),frame))

    if detected:
        # draw bounding boxes on frame
        point1 =  (bbox[1],bbox[0])

        point2 = ((bbox[3]+bbox[1]),(bbox[2]+bbox[0]))
        cv2.rectangle(display_frame, point1, point2, (77, 255, 9), 3, 1)
    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 	# Calculate Frames per second (FPS)
    cv2.putText(display_frame, str(np.mean(q)) + " confidence", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                     display_frame)
    #cv2.imshow("Tracking", cv2.resize(display_frame, (int(3*im_width),int(3*im_height)), interpolation = cv2.INTER_AREA))
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
	   