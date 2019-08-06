from keras import models
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import os
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
img_rows, img_cols = 28, 28
scores = [] #show improvements with data augmentation and fine tuning
x_train = None
for filename in sorted(glob.glob('dataset/train/*.png'),reverse=True, key=os.path.getsize):
    im=cv2.imread(filename)
    im = im[:,:,0].reshape(1, img_rows, img_cols)
    if x_train is not None:
        x_train = np.concatenate((x_train,im), axis=0)
    else:
        x_train = im

x_test = None
for filename in sorted(glob.glob('dataset/test/*.png'),reverse=True, key=os.path.getsize):
    im=cv2.imread(filename)
    im = im[:,:,0].reshape(1, img_rows, img_cols)
    if x_test is not None:
        x_test = np.concatenate((x_test,im), axis=0)
    else:
        x_test = im
        x_test = x_test.reshape(1, img_rows, img_cols)


y_train = np.loadtxt('dataset/train/labels.txt')
y_test = np.loadtxt('dataset/test/labels.txt')
retval, x_train = cv2.threshold(x_train, 1, 255,cv2.THRESH_BINARY)
retval, x_test = cv2.threshold(x_test, 1, 255,cv2.THRESH_BINARY)

x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)


model = models.load_model("models/MNIST_model.h5")
scores.append(model.evaluate(x_test, y_test, verbose=0))
train_datagen = ImageDataGenerator(
      #rescale=1./255,
      width_shift_range=0.1,
      height_shift_range=0.1,
      #zoom_range=[0.5,1.0],
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
train_batchsize = 100
val_batchsize = 10
 
train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=train_batchsize)
 
validation_generator = validation_datagen.flow(
        x_test, y_test,
        batch_size=val_batchsize,
        shuffle=False)

#for layer in model.layers[:1]:
#    layer.trainable = False  # fine tune layers after first
model.layers[0].trainable = False

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=x_train.shape[0]/train_generator.batch_size ,
      epochs=1000,
      validation_data=validation_generator,
      validation_steps=x_test.shape[0]/validation_generator.batch_size,
      verbose=1)
scores.append(model.evaluate(x_test, y_test, verbose=0))


#score = model.evaluate(x_test, y_test, verbose=0)
print('Pre Fine Tuning Performance', scores[0])
print('Post Fine Tuning Performance:', scores[1])
model.save("models/Fine_Tuned.h5")