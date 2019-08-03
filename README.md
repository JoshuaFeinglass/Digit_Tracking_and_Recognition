# Motivations
The MNIST dataset contains handwritten digits 0 through 9 and is frequently used to test proposed methods. Finger movements like writing are very precise as opposed to hand gestures which have a higher variance between examples and users. This project will show how knowledge from a easier problem, like recognizing handwritten digits, can be leveraged to tackle a more difficult one, like recognizing digits drawn through hand gestures.

# Assumptions
In model design, assumptions contrain a problem space to make it easier to solve, but might reduce the number of potential use cases or generalizability of the end product.

In general, I assume that the movement direction is not important for the final classification (only all the pixels that were visited over the action).

For my baseline model, I assume a digit hand gesture takes place over the course of 5 seconds in order to segment actions in time.

# Design Architecture
A hand's presence will be detected in front of the onboard camera. To produce a hand gesture example, the hand will be tracked over a period time and a binary mask will be generated corresponding to all the locations visited by the hand's centroid. A Supervised model initially trained on handwritten digit binary masks will be used to classify the the hand gestures as digits, 0 through 9.

# Methods

## Supervised Model
I use the out of the keras MNIST Convolutional Neural Network. It is theorized that Convolutional Layers, much like a Wavelet Transform, are able to learn primitive features of the dataset like edges vectors, while the Neural Network layers learn more complex features, like the outline of a face.


## Generalization Techniques Utilized
K-Fold Cross Validation:
This technique splits the training data into K equally sized groups. K-1 of the groups are used for training and one is used for validation testing.

Transformation Invariance through Data Augmentation: Strictly supervised models only know what they're told from the training data. If you want the model to be able to recognize something regardless of transformations like translation and scaling, one option is to augment your dataset by performing these transformations on you training data. 

Fine Tuning:


## Hand Detection and Tracking