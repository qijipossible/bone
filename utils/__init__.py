#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from cv2 import *
import numpy


def show(*show_image):
    #for i in range(len(show_image)):
        #cv2.namedWindow(str(i), flags=cv2.WINDOW_KEEPRATIO)
    for i in range(len(show_image)):
        cv2.imshow(str(i), cv2.resize(show_image[i], (1000,1000)))
    cv2.waitKey(0)
