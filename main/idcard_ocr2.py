import util.opencvUtil as opencvUtil
import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
from skimage import io,draw,transform,color
import numpy as np
import json
import pytesseract

pic_path = "../data/idcard_pic/"
import logging
logger = logging.getLogger("身份证图片识别")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

def findTextRegion(img):
    wordInfo = {}



    return wordInfo


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(pic_path + "/gray.jpg", gray)
    logger.info("已生成灰度图【%s】", pic_path + "/gray.jpg")

    # 2. 高斯滤波
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. 自适应二值化方法
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    cv2.imwrite(pic_path + "/binary.png", binary)
    # 4. canny边缘检测
    edged = cv2.Canny(binary, 10, 100)
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[0]
    docCnt = None
    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 对排序后的轮廓循环处理
        for c in cnts:
            # 获取近似的轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
            if len(approx) == 4:
                docCnt = approx
                break
    newimage = img.copy()
    for i in docCnt:
        # circle函数为在图像上作图，新建了一个图像用来演示四角选取
        cv2.circle(newimage, (i[0][0], i[0][1]), 50, (255, 0, 0), -1)
    cv2.imwrite(pic_path + "/newimage.png", newimage)
    # paper = four_point_transform(image, docCnt.reshape(4, 2))
    # 5.根据4个角的坐标值裁剪图片
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    cv2.imwrite(pic_path + "/warped.png", warped)

    # 7. 划分文字区域
    regions = findTextRegion(warped)
    wordInfos = []
    for reg in regions:
        x = reg['x']
        y = reg['y']
        w = reg['w']
        h = reg['h']
        cropImg = gray[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropImg, lang='chi_sim')
        if text == '':
            continue
        cv2.imwrite(pic_path + "/" + text +".png", cropImg)
        word_info = getInfo(x,y,w,h,text)
        logger.info("备注为【%s】,坐标为【%s】",text,word_info['pos'])
        wordInfos.append(word_info)
    cv2.waitKey(0)
    return wordInfos


def four_point_transform(image, docCnt):
    # 自定义
    x1,x2 = max(docCnt[0][0],docCnt[1][0]),min(docCnt[2][0],docCnt[3][0])
    y1,y2 = max(docCnt[0][1],docCnt[3][1]),min(docCnt[1][1],docCnt[2][1])
    cut_img = image[y1:y2, x1:x2]
    # opencv
    # x, y, w, h = cv2.boundingRect(docCnt)
    # cut_img = image[y:y + h, x:x + w]
    return cut_img


# 根据坐标和备注生成wordinfo对象
def getInfo(x,y,w,h,text):
    word_info = {}
    word_info['word'] = text
    pos = []
    pos1 = {}
    pos1['x'] = x
    pos1['y'] = y
    pos2 = {}
    pos2['x'] = x + w
    pos2['y'] = y
    pos3 = {}
    pos3['x'] = x + w
    pos3['y'] = y + h
    pos4 = {}
    pos4['x'] = x
    pos4['y'] = y + h
    pos.append(pos1)
    pos.append(pos2)
    pos.append(pos3)
    pos.append(pos4)
    word_info['pos'] = pos
    return word_info

if __name__ == '__main__':
    init_logger()
    # 读取文件
    img_name = "yh"
    imagePath = "../data/idcard/" + img_name + ".jpg"
    logger.info("图片【%s】识别开始",imagePath)
    pic_path = pic_path + img_name
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    img = cv2.imread(imagePath)
    wordInfos = detect(img)
    label_file = open(pic_path + "/idcard.txt", "w")
    label_file.write(json.dumps(wordInfos,ensure_ascii=False,indent=4))
    logger.info("识别完成，已生成【%s】",pic_path + "/idcard.txt")