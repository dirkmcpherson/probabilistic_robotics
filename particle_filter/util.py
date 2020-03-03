import cv2 as cv
import numpy as np

def scaleImage(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return resized

def drawEllipse(img, pos):
    w = 5
    h = 5
    size = 3
    theta = 0
    pos = (int(pos[0]), int(pos[1])) # ellipse cant handle floats
    cv.ellipse(img, pos, (w,h), theta, 0, 360, 255, thickness=size)

def drawCircle(img, pos, color=(255,255,255)):
    r = 4
    pos = (int(pos[0]), int(pos[1])) # ellipse cant handle floats
    cv.circle(img, pos, r, color, -1)

def dp(p0, p1):
    return (p0[0] - p1[0], p0[1] - p1[1])

def distance(p0, p1):
    dx, dy = dp(p0,p1)
    return np.sqrt(dx**2 + dy**2)
