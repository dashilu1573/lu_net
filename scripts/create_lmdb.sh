#!/usr/bin/en sh
DATA=../data/images
rm -rf $DATA/img_train_lmdb
build/tools/convert_imageset --shuffle \
--resize_height=256 --resize_width=256 \
../data/images/ $DATA/train.txt  $DATA/img_train_lmdb