import tensorflow as tf
import input_loader
import os
import cv2
import numpy as np
import time
from tensorflow.python.ops.variables import trainable_variables
from operator import itemgetter
from skimage.feature import peak_local_max


cell_width = 20
cell_height = 40


def weight_variable(shape, dev):
    initial = tf.truncated_normal(shape, stddev=dev)
    return tf.Variable(initial)


def bias_variable(shape, val):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def detect(x_frame, frame_width, frame_height, frame_width_out, frame_height_out):

    x_tensor = tf.reshape(x_frame, [1, frame_height, frame_width, 3])

    with tf.variable_scope('conv1') as scope:
        w_conv1 = weight_variable([5, 5, 3, 64], 5e-2)
        b_conv1 = bias_variable([64], 0.0)

        h_conv1 = tf.nn.relu(conv2d(x_tensor, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.variable_scope('conv2') as scope:
        w_conv2 = weight_variable([5, 5, 64, 64], 5e-2)
        b_conv2 = bias_variable([64], 0.0)

        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2)

        norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        h_pool2 = max_pool_2x2(norm2)

    # local3
    w_fc1 = weight_variable([int(cell_width * cell_height * 64 / 16), 384], 0.04)
    w_fc1_nonflat = tf.reshape(w_fc1, [int(cell_height / 4), int(cell_width / 4), 64, 384])
    b_fc1 = bias_variable([384], 0.1)

    h_fc1 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_fc1_nonflat, strides=[1, 1, 1, 1], padding='VALID') + b_fc1)

    # local4
    w_fc2 = weight_variable([384, 192], 0.04)
    w_fc2_nonflat = tf.reshape(w_fc2, [1, 1, 384, 192])
    b_fc2 = bias_variable([192], 0.1)

    h_fc2 = tf.nn.relu(tf.nn.conv2d(h_fc1, w_fc2_nonflat, strides=[1, 1, 1, 1], padding='VALID') + b_fc2)

    w_fc3 = weight_variable([192, 2], 1 / 192.0)
    w_fc3_nonflat = tf.reshape(w_fc3, [1, 1, 192, 2])
    b_fc3 = bias_variable([2], 0.0)

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


# sess.run(tf.initialize_variables([frame_height, frame_width]))


kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

##light_poses, capture, empty_frames = input_loader.init()
##
##batch_x, batch_y = input_loader.loadRandomBatch((frame_width/cell_width)*(frame_height/cell_height), cell_width, light_poses, capture, empty_frames)
##frame = np.empty((frame_height, frame_width,3), dtype=np.uint8)
##
##for c_x in range(frame_width/cell_width):
##    for c_y in range(frame_height/cell_height):
##        x_start = c_x * cell_width
##        y_start = c_y*cell_height
##        tmp = batch_x[c_y*(frame_width/cell_width)+c_x,:,:,:]
##        frame[y_start:y_start+cell_height, x_start:x_start+cell_width,:] = tmp
##
##output = sess.run(y, feed_dict={x: frame})
##
##lights = cv2.compare(output[:,:,0], output[:,:,1], cv2.CMP_GT)
##
##lights = cv2.dilate(lights, kernel_5x5)
##for i in range(2):
##    lights = cv2.pyrUp(lights)
##
##out_h = lights.shape[0]
##out_w = lights.shape[1]
##offset_x = (frame_width-out_w)/2
##offset_y = (frame_height-out_h)/2
##
##b, g, r = cv2.split(frame)
##r[offset_y:offset_y+out_h, offset_x:offset_x+out_w] = cv2.max(r[offset_y:offset_y+out_h, offset_x:offset_x+out_w], lights)
##cv2.imshow('R', r)
##
##frame = cv2.merge((b, g, r))
##
##cv2.imshow('Frame', frame)
##cv2.waitKey()


video_file_name = 'input/video1.mp4'
print('Loading file ' + video_file_name + '...')
capture = cv2.VideoCapture(video_file_name)
if not capture.isOpened():
    print('Unable to open ', video_file_name)
    sys.exit()
print('Capture opened')
capture.set(cv2.CAP_PROP_POS_FRAMES, 34100)
ret, frame = capture.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videowriter = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(frame.shape[1]), int(frame.shape[0])))

list_scales = [1.0, 1.5, 2.0, 2.5, 3.0]

while True:
    ret, frame = capture.read()
    frame_out = frame.copy()
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

        start = time.time()
        output = sess.run(y, feed_dict={x: frame_scaled, frame_width: frame_scaled_width,
                                        frame_height: frame_scaled_height,
                                        frame_width_out: np.ceil(frame_scaled_width/4 - 4),
                                        frame_height_out: np.ceil(frame_scaled_height/4 - 9)})
        end = time.time()
        print('Time of running the session: ' + str(end - start))

        lights = peak_local_max(output[:,:,0], min_distance=int(frame_scaled_width/40), threshold_abs=0.7)
        for light in lights:
            result = light[1] * 4 * scale, light[0] * 4 * scale, output[light[0], light[1], 0], scale
            results.append(result)
            print('x(' + str(light[1] * 4 * scale) + '), y(' + str(light[0] * 4 * scale) + '): ' + str(output[light[0], light[1], 0]))
        # lights = cv2.compare(output[:, :, 0], output[:, :, 1], cv2.CMP_GT)
        # cv2.imshow("Lights", lights)
        # cv2.waitKey(0)
        #
        # threshold = 0.5
        # locations = np.where(output[:, :, 0] > threshold)
        # for loc in zip(*locations[::-1]):
        #     print('x(' + str(loc[0]) + '), y(' + str(loc[1]) + '): ' + str(output[loc[1], loc[0], :]))

        # cv2.imshow('Lights', lights)

        # lights = cv2.dilate(lights, kernel_5x5)

        # im_con1, contours1, hierarchy1 = cv2.findContours(lights, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for contour in contours1:
        #     M = cv2.moments(contour)
        #     cX = int(M["m10"] / M["m00"])  # + offset_x
        #     cY = int(M["m01"] / M["m00"])  # + offset_y
        #     print('CONTOURS   x(' + str(cX) + '), y(' + str(cY) + '): ' + str(output[cY, cX, :]))
        #     result = cX * 4 * scale, cY * 4 * scale, output[cY, cX, 0], scale
        #     results.append(result)
        # Resizing to initial shape of frame
        # for i in range(2):
        #     lights = cv2.pyrUp(lights)
        #
        # out_h = lights.shape[0]
        # out_w = lights.shape[1]
        # offset_x = int((frame.shape[1] - out_w) / 2)
        # offset_y = int((frame.shape[0] - out_h) / 2)
        #
        # im_con, contours, hierarchy = cv2.findContours(lights, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #
        # for contour in contours:
        #     M = cv2.moments(contour)
        #     cX = int(M["m10"] / M["m00"] * scale)  # + offset_x
        #     cY = int(M["m01"] / M["m00"] * scale)  # + offset_y
        #     cv2.rectangle(frame_out, (cX, cY), (cX + int(cell_width*scale), cY + int(cell_height*scale)), (0, 255, 0))

        # b, g, r = cv2.split(frame)
        # r[offset_y:offset_y+out_h, offset_x:offset_x+out_w] = cv2.max(r[offset_y:offset_y+out_h, offset_x:offset_x+out_w], lights)
        # frame = cv2.merge((b, g, r))
    results = sorted(results, key=itemgetter(0))
    print(results)

    show_all_scales = True
    # TODO
    if len(results) != 0:
        x_r = int(results[0][0])
        y_r = int(results[0][1])
        prob = results[0][2]
        sc = results[0][3]
    for i, res in enumerate(results):
        if show_all_scales:
            x_r = int(res[0])
            y_r = int(res[1])
            prob = res[2]
            sc = res[3]
            cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                                      (0, 255, 0))
            print('X: ' + str(x_r) + ' Y: ' + str(y_r) + ' Result: ' + str(prob))
        else:
            if res[0] <= x_r+50 and res[1]<= y_r +50 and res[1]>= y_r -50:
                if res[2] >= prob:
                    x_r = int(res[0])
                    y_r = int(res[1])
                    prob = res[2]
                    sc = res[3]
                    if i == len(results) - 1:
                        cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                                      (0, 255, 0))
                        print('X: ' + str(x_r) + ' Y: ' + str(y_r) + ' Result: ' + str(prob))
                else:
                    if i == len(results) - 1:
                        cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                                      (0, 255, 0))
                        print('X: ' + str(x_r) + ' Y: ' + str(y_r) + ' Result: ' + str(prob))
            else:
                cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                              (0, 255, 0))
                print('X: ' + str(x_r) + ' Y: ' + str(y_r) + ' Result: ' + str(prob))
                x_r = int(res[0])
                y_r = int(res[1])
                prob = res[2]
                sc = res[3]
                if i == len(results) - 1:
                    cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                                  (0, 255, 0))
                    print('X: ' + str(x_r) + ' Y: ' + str(y_r) + ' Result: ' + str(prob))

    videowriter.write(frame_out)
    cv2.imshow('Frame', frame_out)
    if cv2.waitKey(10) & 0xFF == 27:
        break

videowriter.release()