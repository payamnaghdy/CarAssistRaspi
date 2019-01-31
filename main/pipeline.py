# Visualizations will be shown in the notebook.
#%matplotlib inline
from importlib import reload
import utils; reload(utils)
from utils import *
# Import crucial modules
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import time
from PIL import Image
import os
import glob
from imutils.video import VideoStream
import argparse
import imutils
import sys
import requests
import json

class ModelConfig:
    """
    ModelConfig is a utility class that stores important configuration option about our model
    """

    def __init__(self, model, name, input_img_dimensions, conv_layers_config, fc_output_dims, output_classes,
                 dropout_keep_pct):
        self.model = model
        self.name = name
        self.input_img_dimensions = input_img_dimensions

        # Determines the wxh dimension of filters, the starting depth (increases by x2 at every layer)
        # and how many convolutional layers the network has
        self.conv_filter_size = conv_layers_config[0]
        self.conv_depth_start = conv_layers_config[1]
        self.conv_layers_count = conv_layers_config[2]

        self.fc_output_dims = fc_output_dims
        self.output_classes = output_classes

        # Try with different values for drop out at convolutional and fully connected layers
        self.dropout_conv_keep_pct = dropout_keep_pct[0]
        self.dropout_fc_keep_pct = dropout_keep_pct[1]


class ModelExecutor:
    """
    ModelExecutor is responsible for executing the supplied model
    """

    def __init__(self, model_config, learning_rate=0.001):
        self.model_config = model_config
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with g.name_scope(self.model_config.name) as scope:
                # Create Model operations
                self.create_model_operations()

                # Create a saver to persist the results of execution
                self.saver = tf.train.Saver()

    def create_placeholders(self):
        """
        Defining our placeholder variables:
            - x, y
            - one_hot_y
            - dropout placeholders
        """

        # e.g. 32 * 32 * 3
        input_dims = self.model_config.input_img_dimensions
        self.x = tf.placeholder(tf.float32, (None, input_dims[0], input_dims[1], input_dims[2]),
                                name="{0}_x".format(self.model_config.name))
        self.y = tf.placeholder(tf.int32, (None), name="{0}_y".format(self.model_config.name))
        self.one_hot_y = tf.one_hot(self.y, self.model_config.output_classes)

        self.dropout_placeholder_conv = tf.placeholder(tf.float32)
        self.dropout_placeholder_fc = tf.placeholder(tf.float32)

    def create_model_operations(self):
        """
        Sets up all operations needed to execute run deep learning pipeline
        """

        # First step is to set our x, y, etc
        self.create_placeholders()

        cnn = self.model_config.model

        # Build the network -  TODO: pass the configuration in the future
        self.logits = cnn(self.x, self.model_config, self.dropout_placeholder_conv, self.dropout_placeholder_fc)
        # Obviously, using softmax as the activation function for final layer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        # Combined all the losses across batches
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        # What method do we use to reduce our loss?
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # What do we really do in a training operation then? Answer: we attempt to reduce the loss using our chosen optimizer
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Get the top prediction for model against labels and check whether they match
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        # Compute accuracy at batch level
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # compute what the prediction would be, when we don't have matching label
        self.prediction = tf.argmax(self.logits, 1)
        # Registering our top 5 predictions
        self.top5_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=5, sorted=True, name=None)


    def predict(self, imgs, top_5=False):
        """
        Returns the predictions associated with a bunch of images
        """
        preds = None
        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            if top_5:
                preds = sess.run(self.top5_predictions, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })
            else:
                preds = sess.run(self.prediction, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })

        return preds

    

from tensorflow.contrib.layers import flatten


def EdLeNet(x, mc, dropout_conv_pct, dropout_fc_pct):
    """
    A variant of LeNet created by Yann Le Cun
    The second parameter, which is encapsulates model configuration, enables varying the convolution filter sizes
    as well as the number of fully connected layers and their output dimensions.
    The third and fourth parameters represent dropout placeholders for convolutional and fully connected layers respectively
    """

    # Used for randomly definining weights and biases
    mu = 0
    sigma = 0.1

    prev_conv_layer = x
    conv_depth = mc.conv_depth_start
    conv_input_depth = mc.input_img_dimensions[-1]

    print(
        "[EdLeNet] Building neural network [conv layers={0}, conv filter size={1}, conv start depth={2}, fc layers={3}]".format(
            mc.conv_layers_count, mc.conv_filter_size, conv_depth, len(mc.fc_output_dims)))

    for i in range(0, mc.conv_layers_count):
        # layer depth grows exponentially
        conv_output_depth = conv_depth * (2 ** (i))
        conv_W = tf.Variable(
            tf.truncated_normal(shape=(mc.conv_filter_size, mc.conv_filter_size, conv_input_depth, conv_output_depth),
                                mean=mu, stddev=sigma))
        conv_b = tf.Variable(tf.zeros(conv_output_depth))

        conv_output = tf.nn.conv2d(prev_conv_layer, conv_W, strides=[1, 1, 1, 1], padding='VALID',
                                   name="conv_{0}".format(i)) + conv_b
        conv_output = tf.nn.relu(conv_output, name="conv_{0}_relu".format(i))
        # Traditional max 2x2 pool
        conv_output = tf.nn.max_pool(conv_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        # Apply dropout - even at the conv level
        conv_output = tf.nn.dropout(conv_output, dropout_conv_pct)

        # Setting our loop variables accordingly
        prev_conv_layer = conv_output
        conv_input_depth = conv_output_depth

    # Flatten results of second convolutional layer so that it can be supplied to fully connected layer
    fc0 = flatten(prev_conv_layer)

    # Now creating our fully connected layers
    prev_layer = fc0
    for output_dim in mc.fc_output_dims:
        fcn_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], output_dim),
                                                mean=mu, stddev=sigma))
        fcn_b = tf.Variable(tf.zeros(output_dim))

        prev_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(prev_layer, fcn_W) + fcn_b), dropout_fc_pct)

    # Final layer (Fully Connected)
    fc_final_W = tf.Variable(tf.truncated_normal(shape=(prev_layer.get_shape().as_list()[-1], mc.output_classes),
                                                 mean=mu, stddev=sigma))
    fc_final_b = tf.Variable(tf.zeros(mc.output_classes))
    logits = tf.matmul(prev_layer, fc_final_W) + fc_final_b

    return logits


def get_imgs_from_folder(path, size=(32, 32), grayscale=False):
    """
    Returns a list of images from a folder as a numpy array
    """
    #img_list = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png")]
    imgs = None
    if grayscale:
        imgs = np.empty([1, size[0], size[1]], dtype=np.uint8)
    else:
        imgs = np.empty([1, size[0], size[1], 3], dtype=np.uint8)

    #for i, img_path in enumerate(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    im = np.array(to_grayscale(img)) if grayscale else np.array(img)
    imgs[0] = im

    return imgs


def get_img(crop_img, size = (32, 32), grayscale=False):
    """
    Returns image as a numpy array
    """
    imgs = None
    if grayscale:
        imgs = np.empty([1, size[0], size[1]], dtype=np.uint8)
    else:
        imgs = np.empty([1, size[0], size[1], 3], dtype=np.uint8)

    # for i, img_path in enumerate(path):
    img = Image.fromarray(crop_img).convert('RGB')
    img = img.resize(size)
    im = np.array(to_grayscale(img)) if grayscale else np.array(img)
    imgs[0] = im

    return imgs


def class_ids_to_labels(cids):
    return list(map(lambda cid: sign_names[sign_names["ClassId"] == cid] ["SignName"].values[0],  cids))


def load():
    datasets_path = "datasets/german_traffic_signs/"
    training_file = datasets_path + 'train.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_train_grayscale = np.asarray(list(map(lambda img: to_grayscale(img), X_train)))
    X_train_grayscale_equalized = np.asarray(
        list(map(lambda img: clahe.apply(np.reshape(img, (32, 32))), X_train_grayscale)))
    return X_train_grayscale_equalized


def preprocess(X_train_grayscale_equalized):
    new_img_grayscale_clahe = np.asarray(list(map(lambda img: clahe.apply(to_grayscale(img)), new_imgs)))
    new_img_grayscale_clahe_normalised = normalise_images(new_img_grayscale_clahe, X_train_grayscale_equalized)
    new_img_grayscale_clahe_normalised = np.reshape(new_img_grayscale_clahe_normalised,
                                                    (new_img_grayscale_clahe_normalised.shape[0], 32, 32, 1))
    return new_img_grayscale_clahe_normalised


# Load sign names file
sign_names = pd.read_csv("signnames.csv")
sign_names.set_index("ClassId")
n_classes = 43
size = (32, 32)
mc_3x3 = ModelConfig(EdLeNet, "EdLeNet_Grayscale_CLAHE_Norm_Take-2_3x3_Dropout_0.50",
                             [32, 32, 1], [3, 32, 3], [120, 84], n_classes, [0.6, 0.5])
me_g_clahe_norm_take2_drpt_0_50_3x3 = ModelExecutor(mc_3x3)

#new_imgs_dir = "./single_image/"
clahe = cv2.createCLAHE(tileGridSize=(4, 4), clipLimit=40.0)
X_train_grayscale_equalized = load()
models_path = "./used_model/"

# load OpenCV's Haar cascade for face detection from disk
detector =  cv2.CascadeClassifier('lbpCascade.xml')

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the total number of example faces written to disk
# thus far
print("[INFO] starting video stream...")
vs = VideoStream(src='http://172.20.55.161:4747/video').start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
total = 1
#id = sys.argv[1]
strs = ["0", "50","60","20","10","30","40","70","80","100","120"]
counter = dict.fromkeys(strs,0)
while True:
    frame = vs.read()
    orig = frame.copy()
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # loop over the face detections and draw them on the frame
    crop_img = frame
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = frame[y:y + h, x:x + w]

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if crop_img.shape != (225, 300, 3):

        new_imgs = get_img(crop_img)
        new_img_grayscale_clahe_normalised = preprocess(X_train_grayscale_equalized)

        preds = me_g_clahe_norm_take2_drpt_0_50_3x3.predict(new_img_grayscale_clahe_normalised)
        preds_lbl = class_ids_to_labels(preds)
        predict_str = preds_lbl[0] 
        print(total, predict_str)
        total += 1
        s = predict_str[0:5]
        speed = ""
        if s == "Speed":
            for i in range(len(predict_str)):
                if predict_str[i] == "(":
                    while predict_str[i + 1] != "k":
                        i += 1
                        speed += predict_str[i]
        else:
            speed = "0"
        counter[speed] += 1
        #save and send
        if counter[speed] >= 5:
            url = 'http://127.0.0.1:8000/api-token-auth/'
            data = {'username':'car','password':'car123456'}
            r = requests.post(url,json=data,headers={'Content-Type': 'application/json' })
            if r.status_code != 200:
                raise ApiError('POST /api-token-auth/ {}'.format(r.status_code))
            token = r.json()['token']
            
            url = 'http://127.0.0.1:8000/position/'
            headers = {'Authorization': 'Token '+token}
            r = requests.get(url, headers=headers)
            for item in r.json():
                url='http://127.0.0.1:8000/signs/list'
            
                data = {'name':predict_str,'latitude':item['latitude'],
                'longitude':item['longitude'],'country':item['country'],'county':item['county'],
                'neighbourhood':item['neighbourhood'],'road':item['road'],'speedlimit':speed,
                'is_uploaded':True}
                r = requests.post(url,json=data,headers={'Content-Type': 'application/json',
                 'Authorization': 'Token '+token})
                counter = dict.fromkeys(strs,0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

