import os
import sys
import cv2
import numpy as np
import yaml
import random
from random import randint

height2width_ratio = 2


def getRandomLightPoses(light_poses, size):
    random_lights = []
    for i in range(int(size)):
        rand_idx = randint(0, len(light_poses) - 1)
        random_lights.append(light_poses[rand_idx])
    return random_lights


def getRandomBackgroundPoses(light_poses, capture_len, capture_width, capture_height, empty_frames, size, ):
    random_backgorunds = []
    # get random sizes based on light sizes
    for i in range(int(size)):
        random_background_pose = {}
        rand_idx = randint(0, len(light_poses) - 1)
        random_light = light_poses[rand_idx]
        random_background_pose['width'] = random_light['width']
        random_background_pose['Frame number'] = empty_frames[randint(0, len(empty_frames) - 1)]
        random_background_pose['Color'] = 'None'
        random_background_pose['x'] = randint(0, int(capture_width) - random_background_pose['width'] - 1)
        random_background_pose['y'] = randint(0, int(capture_height) - height2width_ratio * random_background_pose[
            'width'] - 1)
        random_backgorunds.append(random_background_pose)

    return random_backgorunds


def getRandomLightPrerejectorPoses(input_poses, size, img_size):
    random_lights = []
    for i in range(int(size)):
        rand_idx = randint(0, len(input_poses) - 1)
        pose = input_poses[rand_idx]
        avg_x = pose['x'] + pose['width'] / 2
        avg_y = pose['y'] + height2width_ratio * pose['width'] / 2
        pose['width'] = img_size
        pose['x'] = int(img_size / 2 * round(avg_x / (img_size / 2)) + randint(-img_size / 4, img_size / 4))
        if pose['x'] < 0: pose['x'] = 0
        if pose['x'] > 1240: pose['x'] = 1240
        pose['y'] = int(img_size / 2 * round(avg_y / (img_size / 2)) + randint(-img_size / 4, img_size / 4))
        if pose['y'] < 0: pose['y'] = 0
        if pose['y'] > 680: pose['y'] = 680
        random_lights.append(pose)
    return random_lights


def getRandomBackgroundPrerejectorPoses(empty_frames, size, img_size, capture_width, capture_height):
    random_backgorunds = []
    for i in range(int(size)):
        pose = {}
        pose['width'] = img_size
        pose['Frame number'] = empty_frames[randint(0, len(empty_frames) - 1)]
        pose['Color'] = 'None'
        pose['x'] = randint(0, int(capture_width) - img_size - 1)
        pose['y'] = randint(0, int(capture_height) - img_size - 1)
        random_backgorunds.append(pose)
    return random_backgorunds


def getEmptyFrameList(light_poses, capture_len, margin):
    empty_frames = []
    sorted_light_poses = sorted(light_poses, key=lambda pose: pose['Frame number'])

    first_frame = 0
    last_frame = 0
    for idx in range(len(sorted_light_poses)):
        if idx == 0:
            first_frame = 0
            last_frame = sorted_light_poses[idx]['Frame number'] - margin
        else:
            first_frame = sorted_light_poses[idx - 1]['Frame number'] + margin
            last_frame = sorted_light_poses[idx]['Frame number'] - margin

        if last_frame > first_frame:
            empty_frames += range(first_frame, last_frame)
    # Dont forget about frames at the end of capture
    first_frame = sorted_light_poses[-1]['Frame number'] + margin
    last_frame = int(capture_len - 2)
    if last_frame > first_frame:
        empty_frames += range(first_frame, last_frame)

    return empty_frames


def getROIFromVideoFrame(pose, frame):
    x = pose['x']
    y = pose['y']
    width = pose['width']
    height = pose['width']
    roi = frame[y: y + height, x: x + width]

    return roi


def getLightFromVideoFrame(light, frame):
    x = light['x']
    y = light['y']
    width = light['width']
    height = height2width_ratio * width
    pos_random_magnitude = 1 + width / 15

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    # randomly resize and move the label
    while True:
        width_offset = randint(int(-pos_random_magnitude), int(pos_random_magnitude))
        height_offset = randint(int(-pos_random_magnitude), int(pos_random_magnitude))
        x -= width_offset / 2
        width += width_offset
        y -= width_offset / 2
        height += height_offset
        x += randint(int(-pos_random_magnitude), int(pos_random_magnitude))
        y += randint(int(-pos_random_magnitude), int(pos_random_magnitude))
        if x > 0 and y > 0 and x + width + 1 < frame_width and y + height + 1 < frame_height: break

    light_roi = frame[int(y): int(y + height), int(x): int(x + width)]
    return light_roi


def getBackgroundFromVideoFrame(light, frame):
    x = light['x']
    y = light['y']
    width = light['width']
    height = 3 * width

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    backgorund_roi = frame[int(y): int(y + height), int(x): int(x + width)]
    return backgorund_roi


def loadRandomBatch(batch_size, img_width, light_poses, capture, empty_frames, random_lights_num=-1):
    if batch_size < 1: return  # uzupelnic po okresleniu formatu wyjscia

    # open yaml file - ready for only one

    if random_lights_num < 0:
        random_lights_num = batch_size / 2;
    random_light_poses = []
    random_background_num = batch_size - random_lights_num
    random_backgorund_poses = []

    random_light_poses = getRandomLightPoses(light_poses, random_lights_num)
    random_backgorund_poses = getRandomBackgroundPoses(light_poses, \
                                                       capture.get(cv2.CAP_PROP_FRAME_COUNT), \
                                                       capture.get(cv2.CAP_PROP_FRAME_WIDTH), \
                                                       capture.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                                                       empty_frames, random_background_num, )

    # Generate common sorted list
    batch_items = random_light_poses + random_backgorund_poses
    batch_items = sorted(batch_items, key=lambda x: x['Frame number'])

    img_height = height2width_ratio * img_width
    batch_x = np.empty((batch_size, img_height, img_width, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, 2), dtype=np.uint8)

    # print 'Loading data from video...'
    label_lights = np.array([1, 0], dtype=np.uint8)
    label_backgorund = np.array([0, 1], dtype=np.uint8)
    for item_idx in range(len(batch_items)):
        item = batch_items[item_idx]
        frame_idx = item['Frame number']
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret: continue
        item_roi = np.empty((1, 1, 3))
        item_label = np.empty((2), dtype=np.uint8)
        if item['Color'] != 'None':
            item_roi = getLightFromVideoFrame(item, frame)
            item_label = label_lights
        else:
            item_roi = getBackgroundFromVideoFrame(item, frame)
            item_label = label_backgorund
        item_resized = cv2.resize(item_roi, (img_width, img_height))
        batch_x[item_idx, :, :, :] = item_resized
        batch_y[item_idx, :] = item_label

    return batch_x, batch_y


def loadPreselectedBatch(batch_size, img_height, img_width):
    num_backgrounds = batch_size

    batch_x = np.empty((batch_size, img_height, img_width, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, 2), dtype=np.uint8)

    # print 'Loading data from video...'
    label_lights = np.array([1, 0], dtype=np.uint8)
    label_backgorund = np.array([0, 1], dtype=np.uint8)

    backgorund_files = random.sample(os.listdir('FP'), num_backgrounds)

    cntr = 0

    for item in backgorund_files:
        img_name = 'FP/' + item
        img = cv2.imread(img_name)
        batch_x[cntr, :, :, :] = img
        batch_y[cntr, :] = label_backgorund
        cntr += 1

    return batch_x, batch_y


def loadRandomPrerejectorBatch(batch_size, img_size, light_poses, capture, empty_frames):
    if batch_size < 1: return  # uzupelnic po okresleniu formatu wyjscia

    random_lights_num = batch_size / 2
    random_background_num = batch_size - random_lights_num

    random_light_poses = getRandomLightPrerejectorPoses(light_poses, random_lights_num, img_size)
    random_backgorund_poses = getRandomBackgroundPrerejectorPoses(empty_frames, \
                                                                  random_background_num, img_size, \
                                                                  capture.get(cv2.CAP_PROP_FRAME_WIDTH), \
                                                                  capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Generate common sorted list
    batch_items = random_light_poses + random_backgorund_poses
    batch_items = sorted(batch_items, key=lambda x: x['Frame number'])

    batch_x = np.empty((batch_size, img_size, img_size, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, 2), dtype=np.uint8)

    # print 'Loading data from video...'
    label_lights = np.array([1, 0], dtype=np.uint8)
    label_backgorund = np.array([0, 1], dtype=np.uint8)
    for item_idx in range(len(batch_items)):
        item = batch_items[item_idx]
        frame_idx = item['Frame number']
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret: continue
        item_roi = np.empty((1, 1, 3))
        item_label = np.empty((2), dtype=np.uint8)
        if item['Color'] != 'None':
            item_roi = getROIFromVideoFrame(item, frame)
            item_label = label_lights
        else:
            item_roi = getROIFromVideoFrame(item, frame)
            item_label = label_backgorund
        batch_x[item_idx, :, :, :] = item_roi
        batch_y[item_idx, :] = item_label

    return batch_x, batch_y


def init():
    input_dir = 'input'
    for file in os.listdir(input_dir):
        if file.endswith(".yaml"):
            video_name = os.path.splitext(file)[0]
            print('Loading file ' + file + '...')
            light_poses = []
            with open(input_dir + '/' + file, 'r') as stream:
                try:
                    video_map = yaml.load(stream)
                    light_poses = video_map[video_name]
                except yaml.YAMLError as exc:
                    print(exc)
            print('File loaded, contains', len(light_poses), 'light poses')
            video_file_name = input_dir + '/' + video_name + '.mp4'
            print('Loading file ' + video_file_name + '...')
            capture = cv2.VideoCapture(video_file_name)
            if not capture.isOpened():
                print('Unable to open ', video_file_name)
                sys.exit()
            capture_len = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            print('File loaded, contains', capture_len, 'frames')

            print('Generating list of capture frames with no lights')
            # We consider that empty frame is separated
            # from the nearest frame with lights by a margin
            margin = 30
            empty_frames = getEmptyFrameList(light_poses, capture_len, margin)
            print('Found', len(empty_frames), 'empty frames')

            return light_poses, capture, empty_frames

            ##Example usage
            # light_poses, capture, empty_frames = init()
            # print('Generating random batch')
            # batch_x, batch_y = loadRandomPrerejectorBatch(10, 20, light_poses, capture, empty_frames)
            # print('Random batch ready')
            # print(batch_y)
            # view = batch_x.reshape(200, 20, 3)
            # cv2.imshow('View', view)
            # cv2.waitKey()