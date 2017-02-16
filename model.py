"""For training and saving neural network using Keras.

More info in README.md

"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import adam

import tensorflow as tf

import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize


class train_log_error(Exception):
    """Custom Training log error,- happens when argument is not passed."""

    pass

flags = tf.app.flags
FLAGS = flags.FLAGS

# Input arguments

flags.DEFINE_string('train_log', '', "Training CSV File with recorded driving logs e.g. diriving_log.csv")
flags.DEFINE_string('valid_log', '', "Validation CSV File with recorded driving logs e.g. diriving_log.csv")
flags.DEFINE_string('output_model', '', "Directory to save keras model. e.g. /home/user/udacity/")
flags.DEFINE_string('keras_weights', '', "Pretrained wieghts to use for training. Keras h5 file.")
flags.DEFINE_integer('epochs', 10, "Training epoch count.")
flags.DEFINE_integer('batch_size', 256, "Batch size")
flags.DEFINE_integer('dropout', 0, "0 - Disable | 1 - Enable dropout.")

if FLAGS.train_log == '':
    raise train_log_error('Please provide argument : --train_log path_to_driving_logs_file')

print('\n\nNote. Image directory "IMG" must be in same directory where driving_log.csv is.')


class Utils(object):
    """Helping functions."""

    def load_data(self, paths, targets, Path, use_left_right=True):
        """Load images from files.

        Randomly will load one of three possible images for each path - left, center or right.
        For left and right images there will be steering angle adjusted see shift_ang variables.
        It helps model to better learn steering angles.

        Output is same size as paths.
        """

        dataset = []
        labels = []

        for i, t in enumerate(targets):

            if use_left_right:
                i_lrc = np.random.randint(3)
                if (i_lrc == 0):
                    p = paths['left'][i].strip()
                    shift_ang = .18
                if (i_lrc == 1):
                    p = paths['center'][i].strip()
                    shift_ang = 0
                if (i_lrc == 2):
                    p = paths['right'][i].strip()
                    shift_ang = -.18
            else:
                p = paths['center'][i].strip()
                shift_ang = 0

            path = '/'.join(Path.split('/')[0:-1]) + '/IMG/' + p.split('/')[-1]
            if i == 0:
                print('looking for images at {}'.format(path))

            im_side = cv2.imread(path)
            im_side = cv2.cvtColor(im_side, cv2.COLOR_BGR2RGB)

            dataset.append(im_side)
            labels.append(t + shift_ang)

        return np.array(dataset), np.array(labels)

    def resize_im(self, images):
        """Resize give image dataset."""
        new_w = 200
        new_h = 66
        res_images = np.zeros((images.shape[0], new_h, new_w, images.shape[3]), dtype=images.dtype)

        for i in range(len(images)):
            res_images[i] = imresize(images[i], (new_h, new_w))

        return res_images

    def save_keras_model(self, model, path):
        """Save keras model to given path."""
        model.save_weights(path + 'model.h5')

        with open(path + 'model.json', "w") as text_file:
            text_file.write(m.to_json())

        print('\n\nKeras model saved.')

    def random_brightness(self, images):
        """Add random brightness to give image dataset to imitate day/night."""

        for i in range(len(images)):
            image1 = cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV)
            random_bright = .25 + np.random.uniform()
            image1[:, :, 2] = image1[:, :, 2] * random_bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)

            images[i] = image1

        return images

    def random_shifts(self, images, labels, h_shift, v_shift):
        """Add random horizontal/vertical shifts to image dataset to imitate steering away from sides."""

        rows = images[0].shape[0]
        cols = images[0].shape[1]

        for i in range(len(images)):
            horizontal = h_shift * np.random.uniform() - h_shift / 2
            vertical = v_shift * np.random.uniform() - v_shift / 2

            M = np.float32([[1, 0, horizontal], [0, 1, vertical]])

            # change also corresponding lable -> steering angle
            labels[i] = labels[i] + horizontal / h_shift * 2 * .2
            images[i] = cv2.warpAffine(images[i], M, (cols, rows))

        return images, labels

    def cut_rows(self, dataset, up=0, down=0):
        """Remove specific rows from up and down from given image dataset."""
        return dataset[:, 0 + up:160 - down, :, :]

    def random_shadows(self, images):
        """Add random shadows to given image dataset. It helps model to generalize better."""

        for i in range(len(images)):
            top_y = 320 * np.random.uniform()
            top_x = 0
            bot_x = 160
            bot_y = 320 * np.random.uniform()
            image_hls = cv2.cvtColor(images[i], cv2.COLOR_RGB2HLS)
            shadow_mask = 0 * image_hls[:, :, 1]
            X_m = np.mgrid[0:images[i].shape[0], 0:images[i].shape[1]][0]
            Y_m = np.mgrid[0:images[i].shape[0], 0:images[i].shape[1]][1]

            shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

            if np.random.randint(2) == 1:
                random_bright = .5
                cond1 = shadow_mask == 1
                cond0 = shadow_mask == 0
                if np.random.randint(2) == 1:
                    image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
                else:
                    image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
            images[i] = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

        return images

utils = Utils()

# Main Functions #


def prep_dataset(im_paths, steering_angle, path, use_left_right=True):
    """Prepare dataset for keras model."""
    dataset, labels = utils.load_data(im_paths, steering_angle, path, use_left_right)

    dataset = utils.cut_rows(dataset, 32, 25)
    dataset = utils.resize_im(dataset)

    return dataset, labels


def prep_trainingset(im_paths, steering_angle, path):
    """Prepare dataset for training."""
    dataset, labels = prep_dataset(im_paths, steering_angle, path)
    labels = labels.astype(np.float16)

    print('\n\nTraining dataset shape', dataset.shape)
    dataset_size = labels.shape[0]

    return dataset, labels, dataset_size


def train_generator(dataset, labels, batch_size):
    """Training generator."""
    dataset_size = labels.shape[0]

    start = 0
    while True:
        end = start + batch_size

        d = np.copy(dataset)
        l = np.copy(labels)

        if end <= dataset_size:
            d, l = d[start:end], l[start:end]
        else:
            diff = end - dataset_size
            d = np.concatenate((d[start:], d[0:diff]), axis=0)
            l = np.concatenate((l[start:], l[0:diff]), axis=0)
            start = 0

        d, l = utils.random_shifts(d, l, 22, 16)
        d = utils.random_brightness(d)
        d = utils.random_shadows(d)

        start += batch_size
        yield d, l


def model(data):
    """Create keras model.

    Based on : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    """

    dshape = data.shape

    model = Sequential()
    model.add(BatchNormalization(input_shape=(dshape[1], dshape[2], dshape[3])))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    if FLAGS.dropout == 1:
        model.add(Dropout(0.4))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    if FLAGS.dropout == 1:
        model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    return model


def read_driving_log(path):
    """Read driving_log from file."""
    header = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    driving_log = pd.read_csv(path)
    driving_log.columns = header

    steering_angle = driving_log['steering'].as_matrix()
    im_paths = driving_log[['center', 'left', 'right']]

    return steering_angle, im_paths

print('\n\nPreparing dataset/-s, may take awhile')

# Training dataset and generator #
steering_angle, im_paths = read_driving_log(FLAGS.train_log)
dataset, labels, dataset_size = prep_trainingset(im_paths, steering_angle, FLAGS.train_log)
tg = train_generator(dataset, labels, FLAGS.batch_size)

# Valid dataset and generator if valid_log (validation driving log path) is provided #
if FLAGS.valid_log != '':

    val_steering_angle, val_im_paths = read_driving_log(FLAGS.valid_log)
    valid_data, valid_lables = prep_dataset(val_im_paths, val_steering_angle, FLAGS.valid_log, False)

samples, _ = next(tg)  # load random samples necesarry for model creation
m = model(samples)  # creates model

samples_per_epoch = 10240 - 10240 % FLAGS.batch_size
if dataset_size < samples_per_epoch:
    samples_per_epoch = dataset_size - dataset_size % FLAGS.batch_size

print('Samples per epoch : {}'.format(samples_per_epoch))

if FLAGS.keras_weights != '':
    print('Pretrained weights loaded')
    m.load_weights(FLAGS.keras_weights)

try:
    if FLAGS.valid_log != '':
        m.fit_generator(generator=tg, samples_per_epoch=samples_per_epoch, nb_epoch=FLAGS.epochs, verbose=1,
                        validation_data=(valid_data, valid_lables), pickle_safe=True, nb_worker=12)
    else:
        m.fit_generator(generator=tg, samples_per_epoch=samples_per_epoch, nb_epoch=FLAGS.epochs, verbose=1,
                        pickle_safe=True, nb_worker=12)

    # If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        utils.save_keras_model(m, FLAGS.output_model)
except KeyboardInterrupt:
    # Early stopping possible by interruping script. If output_model dir path provided save model. #
    if FLAGS.output_model != '':
        utils.save_keras_model(m, FLAGS.output_model)
