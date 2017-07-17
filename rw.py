#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
from cv2 import *
import os
import pickle
import pre

path = r"D:\lab\data\bone\\"  # 数据根路径
image_path_origin = path + "origin\\"  # 原版路径
image_path_preprocessed = path + "processed\\"  # 预处理路径
image_path_temp = path + "temp\\"
label_file_path = path + "info.csv"  # 标签信息文件路径

with open(label_file_path) as f:
    lines = f.readlines()


# 通用读取，若找到已预处理后的则直接读取，若没找到则先预处理然后保存
def read_one(index):
    if if_preprocessed_exist(index):
        image, label = read_one_pre(index)
    else:
        image, label = read_one_origin(index)
        image = pre.preprocess(image)  # 预处理
        save_one(image, index)
    return image, label


# 在预处理文件夹中读取
def read_one_pre(index):
    image_file_path = image_path_preprocessed + str(index) + ".jpg"
    image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)  # 直接读为灰度图
    if image is None:
        print('Failed to load image file:', image_file_path)
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转黑白
    return image, get_label(index)


# 在原版文件夹中读取
def read_one_origin(index):
    image_file_path = image_path_origin + str(index) + ".jpg"
    image = cv2.imread(image_file_path)
    if image is None:
        print('Failed to load image file:', image_file_path)
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转黑白
    return image, get_label(index)


# 保存一张图片，默认路径是预处理文件夹，默认名称即编号
def save_one(image, index, save_path=image_path_preprocessed):
    if save_path == "temp":
        save_path = image_path_temp
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(save_path + str(index) + ".jpg", image)


# 保存一个变量，默认路径是数据根路径
def save_pickle(data, name, save_path=path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(save_path + name + ".data", 'wb') as file_name:
        pickle.dump(data, file_name)


# 读取一个变量，默认路径是数据根路径
def load_pickle(name, load_path=path):
    if not os.path.exists(load_path):
        print("ERROR: No such dir, fail to load ", path, name)
        sys.exit(1)
    with open(load_path + name + ".data", 'rb') as file_name:
        return pickle.load(file_name)


# 检测是否存在已预处理后的图片
def if_preprocessed_exist(index):
    return os.path.exists(image_path_preprocessed + str(index) + ".jpg")


def get_label(index):
    info = lines[index+1]
    label = float(info.split(",")[3]) * 10  # 这里与CNN中pridiction联动，避免小数精确度造成的偏差
    return label
