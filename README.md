# Instructions
Make sure there is plenty of light in the room (preferable a light behind the camera shining forward) and that there aren't too many edges in the background (including your face/head). Hold your hand in front of the camera for a few seconds until you see a new black and white frame pop up that indicates the design has entered tracking mode. Draw a digit on the black frame (the perpective is reversed in the program so you can draw the same way you would on a sheet of paper). After 150 frames, the image that was fed into the classifier will be shown to you with a prediction label on it.

# Motivations
The MNIST dataset contains handwritten digits 0 through 9 and is frequently used to test proposed methods. Finger movements like writing are very precise as opposed to hand gestures which have a higher variance between examples and users. This project will show how knowledge from a easier problem, like recognizing handwritten digits, can be leveraged to tackle a more difficult one, like recognizing digits drawn through hand gestures.

# Assumptions
In model design, assumptions contrain a problem space to make it easier to solve, but might reduce the number of potential use cases or generalizability of the end product.

I assume that the movement direction is not important for the final classification (only all the pixels that were visited over the action).

I assume a digit hand gesture takes place over the course of 5 seconds in order to segment actions in time.

# Design Architecture
A hand's presence will be detected in front of the onboard camera. To produce a hand gesture example, the hand will be tracked over a period time and a binary mask will be generated corresponding to all the locations visited by the hand's centroid. A Supervised model initially trained on handwritten digit binary masks will be used to classify the the hand gestures as digits, 0 through 9.

# Methods

## Detection
I wanted something that would work well out of the box and tried many codebases on github to that end, most however, did not work. The victordibia DNN which started detecting immediately. There were quite a few false alarms even when trying to detect only one hand though, so I applied a moving average filter to the detection confidence measure and an M of N filter (detection duty cycle measure) after the confidence detection threshold which allowed me to ensure a clear hand was present before I started tracking. This also ensures that design doesn't immediately switch from detection mode to tracking mode since the M of N filter needs to charge-up.

## Tracking
After trying to do some KLT tracking (estimation of an affine transformation from keypoint homography between frames) from scratch, I ended up using the opencv_contrib Median Flow tracker to track a bounding box of the hand and used the centroid of the bounding box to draw the digit. If the tracker fails on a frame, the design again attempts to detect by taking the highest confidence detection without a detection threshold to ensure a result will be provided. Points that the centroid visits and their neighbors are recorded over the course of the tracking period and are used to generate a final locations traversed mask.

## Pre-processing Before Prediction



## Supervised Model
I use the out of the keras MNIST Convolutional Neural Network. It is theorized that Convolutional Layers, much like a Wavelet Transform, are able to learn primitive features of the dataset like edges vectors, while the Neural Network layers learn more complex features, like the outline of a face.


## Generalization Techniques Utilized
K-Fold Cross Validation:
This technique splits the training data into K equally sized groups. K-1 of the groups are used for training and one is used for validation testing.

Transformation Invariance through Data Augmentation: Strictly supervised models only know what they're told from the training data. If you want the model to be able to recognize something regardless of transformations like translation and scaling, one option is to augment your dataset by performing these transformations on you training data. 

Fine Tuning:

## References
# Detection
https://github.com/victordibia/handtracking
# Tracking:
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
# Data Augmentation
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# Validation
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

