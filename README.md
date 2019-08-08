# Instructions
## General
Make sure there is plenty of light in the room (preferable a light behind the camera shining forward) and that there aren't too many edges in the background (including your face/head). Run 'python demo.py' in the command line. Hold your hand in front of the camera for a few seconds until you see a new black and white frame pop up that indicates the design has entered tracking mode. Draw a digit on the black frame (the perpective is reversed in the program so you can draw the same way you would on a sheet of paper). After 180 frames (~5 seconds), the image that was fed into the classifier will be shown to you with a prediction label on it.

## For TX 2 Demo (DL COE Project)
Same as above, but run 'python3 tx_demo.py'. This version is tailored for the board (I initially developed the project on my computer to be able to work more quickly). This version allows the user to draw for only 90 frames since the frame rate is much slower on the board. Make sure your hand doesn't take up too much space in the frame by sitting back a ways. 

# Motivations
The MNIST dataset contains handwritten digits 0 through 9 and is frequently used to test proposed methods. Finger movements like writing are very precise as opposed to hand gestures which have a higher variance between examples and users. This project will show how two seemlingly dissimilar problems can be framed in way such that knowledge from a easier problem, like recognizing handwritten digits, can be leveraged to tackle a more difficult one, like recognizing digits drawn through hand gestures.

# Design Architecture
A hand's presence will be detected in front of the onboard camera. To produce a hand gesture example, the hand will be tracked over a period time and a binary mask will be generated corresponding to all the locations visited by the hand's centroid. A Supervised model initially trained on handwritten digit binary masks and fine-tuned with generated examples is used to classify the the hand gestures as digits, 0 through 9.

# Dataset
A dataset generated for the purposes of fine tuning can be found under models/train and models/test.

# Methods
### Detection
I wanted something that would work well out of the box and tried many codebases on github to that end, most however, did not work. The victordibia DNN which started detecting immediately. There were quite a few false alarms even when trying to detect only one hand though, so I applied a moving average filter of length 30 to the detection confidence measure and an M of N filter (detection duty cycle measure) after the confidence detection threshold which allowed me to ensure a clear hand was present before I started tracking. This also ensures that design doesn't immediately switch from detection mode to tracking mode since the M of N filter needs to charge-up for at least 20 frames.

### Tracking
After trying to do some KLT tracking (estimation of an affine transformation from keypoint homography between frames) from scratch, I ended up using the opencv_contrib Median Flow tracker to track a bounding box of the hand and used the centroid of the bounding box to draw the digit. If the tracker fails on a frame, the design again attempts to detect by taking the highest confidence detection without a detection threshold to ensure a result will be provided. Points that the centroid visits and their neighbors are recorded over the course of the tracking period and are used to generate a final locations traversed mask.

### Pre-processing
For the purposes of making the hand-written and gesture-written datasets as similar as possible, I thresholded both sets so that any nonzero pixels were 1. After drawing the digit, a developed algorithm in the function bound_binary_image shaves off any zero valued pixels around the digit. I then zero pad the border by a factor of the region of interest's height for the top and bottom and width for the left and right side. Lastly, I resize the image to 28x28x1 before feeding it into the model for prediction. This technique provides translation invariance but has the disadvantage of varying resolutions between the width and height (for example a 1 drawn as only a vertical line might appear to be a 7 since small variations in width would be captured).

### Supervised Model
The code for pre-training the model can be found in train_networth.py

I use the out of the keras MNIST Convolutional Neural Network. Training the network with the provided training and test sets yield an accuracy of 98.5%. Rather than performing hyper-parameter tuning and model comparisons with k-fold cross validation, I opted to use the given model in the interest of time. I trained on the entire MNIST dataset for only 10 epoches with a batch size of 256 (larger batch sizes increase training speed but potentially reduce the accuracy of estimated gradiate) since I fine-tuned the network later. 

<<<<<<< HEAD

### Generalization Techniques Utilized
=======
## Generalization Techniques Utilized
>>>>>>> 64e35cf3d37c05e3e1cb00b2cc668a513b8e7855
The code for Data Augmentation+Fine Tuning can be found in mini_trainer.py.

Transformation Invariance through Data Augmentation: Strictly supervised models only know what they're told from the training data. If you want the model to be able to recognize something regardless of transformations like translation and scaling, one option is to augment your dataset by performing these transformations on you training data. I used the keras ImageDataGenerator class with rotation range 30 degrees, width shift range .1, height shift range .1, and shear range of .5 in order to further explore the input space for each digit without encroaching on another digit (vertically or horizontally flipping the digits would be an example of a poorly chosen transform in this situation).

Fine Tuning: After pretraining a model on a similar problem, fine tuning allows for the model to slightly shift its bias in order to be better suited for the expected test data distribution. In order to reduce training time and ensure the fine tuned model does not diverge too far from the pre-trained model, the weights of the very first convolutional layer are not retrained. This is a common approach since it is generally theorized that the first few layers of a Convolutional Neural Network learn primitive and generally applicable features like edges (similar to a wavelet transform), while later layers learn more complex featurers like the outline of a face. After performing additional training for 100 epoches with an augmented dataset of generated training examples, the model improved from an accuracy of 64.7% on the generated test examples, to 88.2%.

<<<<<<< HEAD
# References and Code Sources
### Detection (demo.py)
=======
## TX2 Version Differences
The length of the filters before and after the NN detector are cut in half to compensate for the slower frame rate on the board. In tracking mode, I simply use the detector without the filters for tracking to avoid installing opencv-contrib (basically the same situation as assuming the median flow tracker always fails). There is more gitter due to the lower frame rate on the board and lack of previous frame information that would be provided from optical flow. To address this, the range of centroid neighbors included in the trace is increased by a factor of 4 (lowering the resolution, but ensuring the trace drawn by the hand remains contiguous).

## References and Code Sources
# Detection (demo.py)
>>>>>>> 64e35cf3d37c05e3e1cb00b2cc668a513b8e7855
https://github.com/victordibia/handtracking
### Tracking (demo.py)
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
### Data Augmentation (mini_trainer.py)
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

