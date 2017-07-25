#!/usr/bin/env bash

DATA=../data/images
echo "create train.txt..."
rm -rf $DATA/train.txt
for i in Black_Footed_Albatross Common_Yellowthroat
do
find $DATA/train -name $i_*.png | cut -d '/' -f2-6 | sed "s/$/ $i/" >> $DATA/train.txt
done

DATA=../data/images
echo "create val.txt..."
rm -rf $DATA/val.txt
for i in Black_Footed_Albatross Common_Yellowthroat
do
find $DATA/val -name $i_*.png | cut -d '/' -f2-6 | sed "s/$/ $i/" >> $DATA/val.txt
done

echo "Done.."