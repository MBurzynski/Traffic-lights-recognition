from __future__ import division
import tensorflow as tf
import cv2
import os
import numpy as np
import input_loader
import time
from tensorflow.python.ops.variables import trainable_variables
from operator import itemgetter
from skimage.feature import peak_local_max


cell_width = 20
cell_height = 40


def weight_variable(shape, dev, name):
    initial = tf.truncated_normal(shape, stddev=dev, name=name)
    return tf.Variable(initial)


def bias_variable(shape, val, name):
    initial = tf.constant(val, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def detect(x_frame, frame_width, frame_height, frame_width_out, frame_height_out):

    x_tensor = tf.reshape(x_frame, [1, frame_height, frame_width, 3], name="input_tensor")

    with tf.variable_scope('conv1') as scope:
        w_conv1 = weight_variable([5, 5, 3, 64], 5e-2, "weight1")
        b_conv1 = bias_variable([64], 0.0, "bias1")

        h_conv1 = tf.nn.relu(conv2d(x_tensor, w_conv1, strides=[1,1,1,1]) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")

    with tf.variable_scope('conv2') as scope:
        w_conv2 = weight_variable([5, 5, 64, 64], 5e-2, "weight2")
        b_conv2 = bias_variable([64], 0.0, "bias2")

        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2)

        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")

        h_pool2 = max_pool_2x2(norm2)

    # local3
    w_fc1 = weight_variable([int(cell_width * cell_height * 64 / 16), 384], 0.04, "weight3")
    w_fc1_nonflat = tf.reshape(w_fc1, [int(cell_height / 4), int(cell_width / 4), 64, 384])
    b_fc1 = bias_variable([384], 0.1, "bias3")

    h_fc1 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_fc1_nonflat, strides=[1, 1, 1, 1], padding='VALID') + b_fc1)

    # local4
    w_fc2 = weight_variable([384, 192], 0.04, "weight4")
    w_fc2_nonflat = tf.reshape(w_fc2, [1, 1, 384, 192])
    b_fc2 = bias_variable([192], 0.1, "bias4")

    h_fc2 = tf.nn.relu(tf.nn.conv2d(h_fc1, w_fc2_nonflat, strides=[1, 1, 1, 1], padding='VALID') + b_fc2)

    w_fc3 = weight_variable([192, 2], 1 / 192.0, "weight5")
    w_fc3_nonflat = tf.reshape(w_fc3, [1, 1, 192, 2])
    b_fc3 = bias_variable([2], 0.0, "bias5")

    y_conv = tf.nn.softmax(tf.nn.conv2d(h_fc2, w_fc3_nonflat, strides=[1, 1, 1, 1], padding='VALID') + b_fc3)

    y_out = tf.reshape(y_conv, [frame_height_out, frame_width_out, 2])

    return y_out


x = tf.placeholder(tf.float32, shape=[None, None, 3])
frame_width = tf.Variable(0, name='f_w', trainable=False)
frame_height = tf.Variable(0, name='f_h', trainable=False)
frame_width_out = tf.Variable(0, name='f_w_o', trainable=False)
frame_height_out = tf.Variable(0, name='f_h_o', trainable=False)
y = detect(x, frame_width, frame_height, frame_width_out, frame_height_out)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver(trainable_variables())

# try to restore model
saver.restore(sess, 'train/model')
print("Model restored.")

light_poses, capture, empty_frames = input_loader.init()
light_poses = sorted(light_poses, key=lambda x: x['Frame number'])

light_poses_zip = []
lights_on_one_frame = []
frame_number = light_poses[0]['Frame number']

for idx in range(len(light_poses)):
    light = light_poses[idx]
    if light['Frame number'] == frame_number:
        lights_on_one_frame.append(light)
    else:
        light_poses_zip.append(lights_on_one_frame)
        lights_on_one_frame = []
        frame_number = light['Frame number']
        lights_on_one_frame.append(light)


video_file_name = 'input/video1.mp4'
print('Loading file ' + video_file_name + '...')
capture = cv2.VideoCapture(video_file_name)
if not capture.isOpened():
    print('Unable to open ', video_file_name)
print('Capture opened')
capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 1)
ret, frame = capture.read()
fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
videowriter = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(frame.shape[1]), int(frame.shape[0])))

output_path_G = 'TP_G'
if not os.path.exists(output_path_G):
    os.mkdir(output_path_G)

output_path_R = 'TP_R'
if not os.path.exists(output_path_R):
    os.mkdir(output_path_R)

output_path_Y = 'TP_Y'
if not os.path.exists(output_path_Y):
    os.mkdir(output_path_Y)

output_path_RY = 'TP_RY'
if not os.path.exists(output_path_RY):
        os.mkdir(output_path_RY)

index_g, index_r, index_y, index_ry = 0, 0, 0, 0

list_scales = [1.0, 1.5, 2.25, 2.75]

for item_idx in range(len(light_poses_zip)):
    # if item_idx < 2890:
    #     continue
    start = time.time()
    item = light_poses_zip[item_idx][0]
    frame_idx = item['Frame number']
    color = item['Color']
    ret, frame = capture.read()
    frame_out = frame.copy()
    frame = frame[0:int(frame.shape[0] * 7 / 10), 0:frame.shape[1]]
    results = []
    if not ret:
        print('Video finished')
        break

    for scale in list_scales:
        frame_scaled_width, frame_scaled_height = int(frame.shape[1]/scale), int(frame.shape[0]/scale)
        if scale == 1.0:
            frame_scaled = frame.copy()
        else:
            frame_scaled = cv2.resize(frame, (frame_scaled_width, frame_scaled_height))

        output = sess.run(y, feed_dict={x: frame_scaled, frame_width: frame_scaled_width,
                                        frame_height: frame_scaled_height,
                                        frame_width_out: np.ceil(frame_scaled_width/4 - 4),
                                        frame_height_out: np.ceil(frame_scaled_height/4 - 9)})

        lights = peak_local_max(output[:,:,0], min_distance=int(frame_scaled_width/85), threshold_abs=0.75)
        for light in lights:
            result = light[1] * 4 * scale, light[0] * 4 * scale, output[light[0], light[1], 0], scale
            results.append(result)
            print('x(' + str(light[1] * 4 * scale) + '), y(' + str(light[0] * 4 * scale) + '): ' + str(output[light[0], light[1], 0]))

    results = sorted(results, key=itemgetter(0))

    for i, res in enumerate(results):
        x_r = int(res[0])
        y_r = int(res[1])
        sc = res[3]
        # cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
        #                           (0, 255, 0))
        for fr in range(len(light_poses_zip[item_idx])):
            item = light_poses_zip[item_idx][fr]
            print('X [real]: ' + str(item['x']) + ' Y [real]: ' + str(item['y']))
            if x_r > item['x'] - 10 - sc*5 and x_r < item['x'] + 10 + sc*5 and y_r > item['y'] - 10 - sc*5 and y_r < item['y'] + 10 + sc*5 and sc*cell_width > 0.8*item['width']:
                light_img = frame[y_r:y_r + int(cell_height * sc), x_r:x_r + int(cell_width * sc)]
                if sc != 1.0:
                    light_img = cv2.resize(light_img, (20, 40))
                if color == 'Red':
                    cv2.imwrite(output_path_R + "/tp_" + str(index_r) + ".png", light_img)
                    index_r += 1
                elif color == 'Green':
                    cv2.imwrite(output_path_G + "/tp_" + str(index_g) + ".png", light_img)
                    index_g += 1
                elif color == 'Yellow':
                    cv2.imwrite(output_path_Y + "/tp_" + str(index_y) + ".png", light_img)
                    index_y += 1
                elif color == 'RedYellow':
                    cv2.imwrite(output_path_RY + "/tp_" + str(index_ry) + ".png", light_img)
                    index_ry += 1
        print('X: ' + str(x_r) + ' Y: ' + str(y_r))

    # videowriter.write(frame_out)
    # cv2.imshow('Frame', frame_out)
    # if cv2.waitKey(10) & 0xFF == 27:
    #     break

    end = time.time()
    print('Time of running the session: ' + str(end - start))
    print('Progress: ' + str(item_idx) + '/' + str(len(light_poses_zip)))

videowriter.release()