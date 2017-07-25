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


def threshold_lift(image, low_thresh, new_val):
    for x in np.nditer(image, op_flags=['readwrite']):
        if x < low_thresh:
            x[...] = new_val
    return image


def find_background_color(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])[:-1, 0]  # 将直方图256个数组成一个list
    last_v = nodown = down = doubt = anchor= 0
    flag = False
    background_color = 60

    peak = hist[1:].argmax() + 1
    next_peak = hist[peak+1:].argmax() + peak+1

    for index, v in enumerate(hist[next_peak:], next_peak):
        if (last_v - v) > 400:
            down += 1
            if down > 5:
                flag = True
            if doubt:
                if v < anchor:
                    doubt = 0
        else:
            nodown += 1
            if flag:
                if not anchor == 0:
                    anchor = last_v
                doubt += 1
                if doubt > 2:
                    if v < 10000:
                        background_color = index - 3
                        break
                    else:
                        down = doubt = anchor = 0
                        flag = False
        last_v = v
    return background_color


def preprocess(origin_image, length=200):
    h, w = origin_image.shape[:2]
    image = origin_image.copy()

    bg_color = find_background_color(image)
    #print(bg_color)
    #plt.hist(image.ravel(), 256, [0, 256])
    #plt.show()

    threshold_lift(image, bg_color, bg_color)
    #lifted = image.copy()
    
    image = cv2.equalizeHist(image)
    #equalized = image.copy()
    mask = threshold_lift(image.copy(), 220, 0)

    threshold_lift(image, bg_color+4, bg_color+4)
    lifted2 = image.copy()

    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    seed_point = (int(w / 2), int(h / 10))  # flood起始点，只要选择一个典型的背景点即可
    cv2.floodFill(image, mask, seed_point, 0, 255, 7)  # 最后一个参数3若换成2，则有些非骨部分去不掉，若换成4则大多骨部分都被去掉
    #utils.show(lifted, equalized, lifted2, image)
    #rw.save_one(image, "temp", "temp")


    # 去噪点，这里利用了轮廓周长来进行降噪，先寻找轮廓，轮廓周长低于某个值的即认为噪点，填充。系数若取205是正好能去除“L”的，但会误伤比较小的骨头。
    image = denoise_by_length(image, 130)

    # 开操作，平滑轮廓，去除过于夸张的细碎枝丫。经过对比发现MORPH_ELLIPSE的平滑效果优于方形和十字。
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))

    #utils.show(image)

    # 超大尺寸预裁剪
    MAX = 2000
    if h > MAX or w > MAX:
        w_over = w - MAX
        l = int(w_over/2)
        image = image[:MAX, l:MAX+l]
        h, w = image.shape[:2]

    # 统一尺寸化（MAX*MAX）
    sub = MAX - w
    left = int(sub/2)
    right = sub - left
    image = cv2.copyMakeBorder(image, MAX-h, 0, left, right, cv2.BORDER_CONSTANT, 0)

    # 缩小
    if length > 1:
        image = cv2.resize(image, (length, length))

    return image


if __name__ == "__main__":
    for i in range(10100):
        im, label = rw.read_one_origin(i)
        im = preprocess(im)
        rw.save_one(im, i, "temp")

"""
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
print(hist)
plt.hist(image.ravel(), 256, [0, 256])
plt.show()thresh_val = np.max(image[0:100, 500:600]) + 30

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