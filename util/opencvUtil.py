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

# 降噪
def noise_remove_cv2(image, k):
    """
    8邻域降噪
    Args:
        image: 图片
        k: 判断阈值

    Returns:

    """

    def calculate_noise_count(img_obj, w, h):
        """
        计算邻域非白色的个数
        Args:
            img_obj: img obj
            w: width
            h: height
        Returns:
            count (int)
        """
        count = 0
        width, height = img_obj.shape
        for _w_ in [w - 1, w, w + 1]:
            for _h_ in [h - 1, h, h + 1]:
                if _w_ > width - 1:
                    continue
                if _h_ > height - 1:
                    continue
                if _w_ == w and _h_ == h:
                    continue
                if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
                    count += 1
        return count

    w, h = image.shape
    for _w in range(w):
        for _h in range(h):
            if _w == 0 or _h == 0:
                image[_w, _h] = 255
                continue
            # 计算邻域pixel值小于255的个数
            pixel = image[_w, _h]
            if pixel == 255:
                continue

            if calculate_noise_count(image, _w, _h) < k:
                image[_w, _h] = 255

    return image