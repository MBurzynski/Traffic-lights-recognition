import tensorflow as tf
import cv2
import numpy as np
import time
from tensorflow.python.ops.variables import trainable_variables
from operator import itemgetter
from skimage.feature import peak_local_max

cell_width = 20
cell_height = 40

lights_svm = cv2.ml.SVM_load("input/LightsDetectorAuto.yml")
color_svm = cv2.ml.SVM_load("input/ColorDetectorAuto.yml")


def compute_hog(img):
    hog = cv2.HOGDescriptor("input/hog.xml")
    locations = []
    features = hog.compute(img, (5, 5), (0, 0), locations)
    return features


def image_to_patches(img, patch_size):
    patches = []
    for i in range(int(img.shape[0]/patch_size)):
        for j in range(int(img.shape[1]/patch_size)):
            patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch)
    return patches


def compute_histograms(img, features):
    patches = image_to_patches(img, 5)
    for patch in patches:
        b_hist = cv2.calcHist([patch], [0], None, [10], [0, 256])
        g_hist = cv2.calcHist([patch], [1], None, [10], [0, 256])
        r_hist = cv2.calcHist([patch], [2], None, [10], [0, 256])

        cv2.normalize(b_hist, b_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(g_hist, g_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(r_hist, r_hist, 0, 1, cv2.NORM_MINMAX)

        features = np.concatenate((features, b_hist))
        features = np.concatenate((features, g_hist))
        features = np.concatenate((features, r_hist))
    return features


def clasify(features):
    features = np.matrix(features)
    features = np.reshape(features, (1, features.size))
    result_light = lights_svm.predict(features)
    result_color = 0
    if result_light[1][0] == 1:
        result_color = color_svm.predict(features)
        result_color = result_color[1][0]
    return result_color


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

    y_out = tf.reshape(y_conv, [frame_height_out, frame_width_out, 2], name="output_node")

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


video_file_name = 'input/video1.mp4'
print('Loading file ' + video_file_name + '...')
capture = cv2.VideoCapture(video_file_name)
if not capture.isOpened():
    print('Unable to open ', video_file_name)
print('Capture opened')
capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
ret, frame = capture.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videowriter = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(frame.shape[1]), int(frame.shape[0])))

list_scales = [1.0, 1.5, 2.25, 2.75]

while True:
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

        start = time.time()
        output = sess.run(y, feed_dict={x: frame_scaled, frame_width: frame_scaled_width,
                                        frame_height: frame_scaled_height,
                                        frame_width_out: np.ceil(frame_scaled_width/4 - 4),
                                        frame_height_out: np.ceil(frame_scaled_height/4 - 9)})
        end = time.time()
        print('Time of running the session: ' + str(end - start))

        lights = peak_local_max(output[:,:,0], min_distance=int(frame_scaled_width/120), threshold_abs=0.75)
        for light in lights:
            result = light[1] * 4 * scale, light[0] * 4 * scale, output[light[0], light[1], 0], scale
            results.append(result)
            print('x(' + str(light[1] * 4 * scale) + '), y(' + str(light[0] * 4 * scale) + '): ' + str(output[light[0], light[1], 0]))

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
            light_img = frame[y_r:y_r + int(cell_height * sc), x_r:x_r + int(cell_width * sc)]
            light_img_gray = cv2.cvtColor(light_img, cv2.COLOR_BGR2GRAY)
            start_hog = time.time()
            features = compute_hog(cv2.resize(light_img_gray, (20, 40)))
            features = compute_histograms(cv2.resize(light_img, (20, 40)), features)
            result_color = clasify(features)
            end_hog = time.time()
            print("Time of SVM prediction: " + str(end_hog-start_hog))
            color = (255, 255, 255)
            if result_color == 1:
                print("Prediction: Red Light")
                color = (0, 0, 255)
            elif result_color == 2:
                print("Prediction: Green Light")
                color = (0, 255, 0)
            elif result_color == 3:
                print("Prediction: Yellow Light")
                color = (0, 255, 255)
            elif result_color == 4:
                print("Prediction: Red-Yellow Light")
                color = (0, 100, 255)
            cv2.rectangle(frame_out, (x_r, y_r), (x_r + int(cell_width * sc), y_r + int(cell_height * sc)),
                          color, 2)
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