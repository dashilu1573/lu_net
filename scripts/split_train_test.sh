#!/usr/bin/env bash

# 打乱样本集合
cat raw_data.txt | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' >> data.txt

# 分成训练样本和测试样本,变量赋值等号两边不能有空格
num=$(wc -l data.txt| awk '{print $1}')
echo "样本数量：$num"

train_num=`expr $num / 10 \* 8`
echo "训练样本数量：$train_num"
head -$train_num data.txt > train_data.txt


test_num=`expr $num / 10 \* 2`
echo "测试样本数量：$test_num"
tail -$test_num data.txt > test_data.txt