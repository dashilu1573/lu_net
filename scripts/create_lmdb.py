'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import sys
import glob
import random
import numpy as np
import cv2
import lmdb

sys.path.append("..")
from proto.lu_pb2 import Datum

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


# image is numpy.ndarray format. BGR instead of RGB
def make_datum(img, label):
    return Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_list_file = '../data/images/train.txt'
train_images_root = '../data/images/train'
test_list_file = '../data/images/test.txt'
test_images_root = '../data/images/test'

train_lmdb = '../data/images/train_lmdb'
validation_lmdb = '../data/images/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


print 'Creating train_lmdb\n'
f = open(train_list_file, 'r')
trainlist = f.readlines()
f.close()

# Shuffle train_data
random.shuffle(trainlist)

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path_label in enumerate(trainlist):
        (img_path, label) = img_path_label.split()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        datum = make_datum(img, label)

        # key:line index, value:image data +label
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print 'Creating validation_lmdb\n'
f = open(test_list_file, 'r')
testlist = f.readlines()
f.close()

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path_label in enumerate(testlist):
        (img_path, label) = img_path_label.split()

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        datum = make_datum(img, label)

        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print 'Finished processing all images\n'
