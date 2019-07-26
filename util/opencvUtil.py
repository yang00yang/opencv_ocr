import cv2
import numpy as np

# 灰度处理
def grayImg(image):
    # 对图像进行处理，转化为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

# 二值化处理
def binary(image,threshod):
    # 二值化
    retval, binary = cv2.threshold(image, threshod, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return binary

# 膨胀
def dilate(img,size):
    #腐蚀
    kernel = np.ones((size, size), np.uint8)
    dilation = cv2.dilate(img, kernel)  # 膨胀
    return dilation

# 腐蚀
def erode(img,size):
    #腐蚀
    kernel = np.ones((size, size), np.uint8)
    erosion = cv2.erode(img, kernel)  # 腐蚀
    return erosion

# 边缘检测
def carry(img):
    edges = cv2.Canny(img, 100, 200)
    return edges

# 轮廓检测
def findContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 清除小面积轮廓
def drawContours(img,contours):
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    return img

# 连通域
def connectedComponents(img):
    ret, labels = cv2.connectedComponents(img, connectivity=None)
    return labels