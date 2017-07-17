#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rw, utils, CNN

# 请先在rw.py中设置数据目录

"""
train_x = list()
train_y = list()
test_x = list()
test_y = list()

print("preparing train data")
for i in range(10000):

    image, label = rw.read_one(i)  # 若已经存在i的预处理后版本则直接读取预处理后版本，若无则进行预处理并保存预处理后文件。若想强制预处理则需手动删除已保存的预处理后文件。

    train_x.append(image)
    train_y.append((label,))

print("preparing test data")
for i in range(10000, 10100):
    image, label = rw.read_one(i)

    test_x.append(image)
    test_y.append((label,))

train_data = {"x": train_x, "y": train_y}
test_data = {"x": test_x, "y": test_y}

rw.save_pickle([train_data, test_data], "0-10000_10000-10100")
"""
data = rw.load_pickle("0-10000_10000-10100")
train_data = data[0]
test_data = data[1]


CNN.train_model(train_data, test_data)
