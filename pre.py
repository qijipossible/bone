#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from cv2 import *
import numpy as np
from matplotlib import pyplot as plt
import rw, utils
import sys


def denoise_by_length(image_origin, length):
    image = image_origin.copy()
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cnts = [cnt for cnt in cnts if cv2.arcLength(cnt, True) < length]
    cv2.drawContours(image, cnts, -1, 0, cv2.FILLED)
    return image


def threshold_lift(image, lowThresh, newVal):
    for x in np.nditer(image, op_flags=['readwrite']):
        if x < lowThresh:
            x[...] = newVal


def find_background_color(image):
    pass


def preprocess(origin_image, length=200):
    h, w = origin_image.shape[:2]
    image = origin_image.copy()

    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

    threshold_lift(image, 77, 77)
    lifted = image.copy()
    image = cv2.equalizeHist(image)
    equalized = image.copy()
    threshold_lift(image, 77, 77)
    seed_point = (int(w / 2), int(h / 10))  # flood起始点，只要选择一个典型的背景点即可
    #seed_color = int(np.round(np.mean(image[int(h / 10)-3:int(h / 10)+3, int(w / 2)-3:int(w / 2)+3:]))) + 10
    cv2.floodFill(image, None, seed_point, 0, 255, 7)  # 最后一个参数3若换成2，则有些非骨部分去不掉，若换成4则大多骨部分都被去掉
    utils.show(lifted, equalized, image)
    #rw.save_one(image, "temp", "temp")

    #hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    #print(hist)
    #plt.hist(image.ravel(), 256, [0, 256])
    #plt.show()

    # 去噪点，这里利用了轮廓周长来进行降噪，先寻找轮廓，轮廓周长低于某个值的即认为噪点，填充。这里的40是测试值。另外似乎填充也有问题，太大块的并不能填充完全，还留有中心。
    image = denoise_by_length(image, 250)

    # 开操作，平滑轮廓，去除过于夸张的细碎枝丫。经过对比发现MORPH_ELLIPSE的平滑效果优于方形和十字。
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

    utils.show(image)

    # 方形化
    if h > w:
        sub = h - w
        left = int(sub/2)
        right = sub - left
        image = cv2.copyMakeBorder(image, 0,0,left,right, cv2.BORDER_CONSTANT, 0)

    # 缩小
    if length > 1:
        image = cv2.resize(image, (length, length))

    return image


if __name__ == "__main__":
    image, label = rw.read_one_origin(3)

    preprocess(image)

"""thresh_val = np.max(image[0:100, 500:600]) + 30
thresh = cv2.threshold(image, thresh_val, 255, cv2.THRESH_TOZERO)[1]
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
eroded = cv2.erode(flooded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

#kernel = np.array([[1/8,1/8,1/8], [1/8,0,1/8], [1/8,1/8,1/8]])
#kernel = np.ones([5,5])
#kernel[3,3] = -24
#denoised = cv2.filter2D(preflooded, -1, kernel)

#cnts = cv2.findContours(denoised.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[1]
#cv2.drawContours(denoised, cnts, -1, 255)

#eroded2 = cv2.erode(denoised, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
#dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
"""