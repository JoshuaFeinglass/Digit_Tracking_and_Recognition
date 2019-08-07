from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
from keras import models
import cv2
import sys
import numpy as np
from collections import deque
def bound_binary_image(img,w,h):
    i=0
    x0_bound = None
    x1_bound = None
    y0_bound = None
    y1_bound = None
    while not(x0_bound and x1_bound and y0_bound and y1_bound):
        x0 = i
        x1 = w-i-1
        y0 = i
        y1 = h-i-1
        if not x0_bound and np.any(img[:,int(x0)]):
            x0_bound=x0
        if not x1_bound and np.any(img[:,int(x1)]):
            x1_bound=x1
        if not y0_bound and np.any(img[int(y0),:]):
            y0_bound=y0
        if not y1_bound and np.any(img[int(y1),:]):
            y1_bound=y1
        i+=1
    return (y0_bound,y1_bound,x0_bound,x1_bound)

def convert_to_tracking_format(detect,ind):
    (left, right, top, bottom) = (detect[ind][1] * im_width, detect[ind][3] * im_width,
                                  detect[ind][0] * im_height, detect[ind][2] * im_height)
    
    bounding_box = (int(left),int(top),int(right-left),int(bottom-top))
    return bounding_box

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

#initialize before loop
k = cv2.waitKey(1) & 0xff
detection_graph, sess = detector_utils.load_inference_graph()
digit_model = models.load_model("models/Fine_Tuned.h5")
detected = 0

M = 20
N = 30
taps = 30
num_detects = deque(maxlen=N) #M of N filter
q = deque(maxlen=taps) #rolling average filter
j = 0
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
        if np.sum(num_detects) > M: #duty cycle from M of N
            bbox = convert_to_tracking_format(boxes,0)#np.argmax(scores))
            path_mask = np.zeros_like(frame)
            path_mask = path_mask[:,:,0]
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
        path_mask[cY - 5:cY + 5, cX - 5:cX + 5] = 255
        cv2.circle(display_frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.imshow("path",path_mask)
        loop_num += 1
        cv2.putText(display_frame, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        loop_num+=1
        if loop_num >= 180:
            detected = 0
            num_detects = deque(maxlen=N) #M of N filter
            q = deque(maxlen=taps) #rolling average filter
            ret, path_mask = cv2.threshold(path_mask, 127, 255, cv2.THRESH_BINARY)
            bounding = bound_binary_image(path_mask,im_width,im_height)
            ROI = path_mask[int(bounding[0]):int(bounding[1]), int(bounding[2]):int(bounding[3])]
            x_border = int((bounding[3]-bounding[2])/5)
            y_border = int((bounding[3]-bounding[2])/5)
            ROI_padded = cv2.copyMakeBorder(ROI,y_border,y_border,x_border,x_border,cv2.BORDER_CONSTANT, 0)
            model_input = cv2.resize(ROI_padded, dsize=(28, 28))
            cv2.imwrite("dataset/qiao/"+str(j)+".png",model_input)
            j+=1
            prediction = digit_model.predict_classes(model_input.reshape(1,28,28,1))
            model_input = cv2.resize(model_input, (int(20 * 28), int(20 * 28)), interpolation=cv2.INTER_AREA)
            cv2.putText(model_input,"Prediction: "+str(prediction), (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 10), 2)
            cv2.imshow("path",model_input) #int(bounding[3]):int(bounding[2])])

    if detected:
        # draw bounding boxes on frame
        point1 = (int(bbox[0]),int(bbox[1]))

        point2 = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
        cv2.rectangle(display_frame, point1, point2, (77, 255, 9), 3, 1)
    cv2.putText(display_frame, str(np.mean(q)) + " confidence", (150,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                     display_frame)
    cv2.imshow("Tracking", cv2.resize(display_frame, (int(3*im_width),int(3*im_height)), interpolation = cv2.INTER_AREA))
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break