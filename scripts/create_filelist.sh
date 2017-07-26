#!/usr/bin/env bash

DATA=../data/cifar-10
echo "create train.txt..."
rm -rf $DATA/train.txt
for i in 0 1 2 3 4 5 6 7 8 9
do
find $DATA/train -name $i_*.jpg | cut -d '/' -f2-6 | sed "s/$/ $i/" >> $DATA/train.txt
done

DATA=../data/cifar-10
echo "create val.txt..."
rm -rf $DATA/val.txt
for i in 0 1 2 3 4 5 6 7 8 9
do
find $DATA/val -name $i_*.jpg | cut -d '/' -f2-6 | sed "s/$/ $i/" >> $DATA/val.txt
done

echo "Done.."