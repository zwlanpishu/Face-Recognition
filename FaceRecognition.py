from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks import *

np.set_printoptions(threshold = np.nan)

FRmodel = faceRecoModel(input_shape = (3, 96, 96))
print("Total Params: ", FRmodel.count_params())


"""
Define the triplet cost function
"""
def triplet_loss(y_true, y_pred, alpha = 0.2) :
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # compute the distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(positive, anchor)), axis = -1)

    # compute the distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(negative, anchor)), axis = -1)

    # compute the loss of one training sample
    loss = pos_dist - neg_dist + alpha

    # compute the cost over all training samples
    cost = tf.reduce_sum(tf.maximum(loss, 0))

    return cost


"""
Load the trained model
"""
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)


"""
Build a database that include all the people allowed
This database maps each person's name to a 128-dimensional encoding of their face
"""
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

"""
Verify if the person is the one claimed by id card 
"""
def verify(image_path, identity, database, model) : 
    
    # compute the encoding of current image token by camera
    encoding = img_to_encoding(image_path, model)

    # compute the distance before current encoding and the one stored in the database
    dist = np.linalg.norm(encoding - database[identity])

    # determine wheather open the door
    if dist < 0.7 : 
        print("It's " + str(identity) + ", welcome！")
        door_open = True
    elif dist >= 0.7 :
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    
    return dist, door_open


verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)

"""
Implement the face recognition function, 
which try to find the person with smallest dist to the image 
"""
def who_is_it(image_path, database, model) : 

    # compute the encoding of current image token by camera
    encoding = img_to_encoding(image_path, model)
    min_dist = 10000

    # look over the database
    for (name, db_enc) in database.items() : 
        dist = np.linalg.norm(encoding - database[name])

        if dist < min_dist : 
            min_dist = dist
            identity = name
    
    # determine wheather open the door
    if min_dist < 0.7 : 
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        door_open = True
    elif min_dist >= 0.7 :
        print("Not in the database")
        door_open = False
    
    return min_dist, door_open


who_is_it("images/camera_0.jpg", database, FRmodel)
who_is_it("images/camera_1.jpg", database, FRmodel)
