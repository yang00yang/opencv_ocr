import util.opencvUtil as opencvUtil
import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
from skimage import io,draw,transform,color
import numpy as np
import json
from util import httpUtil

pic_path = "../data/gjj_pic/"
import logging
logger = logging.getLogger("公积金图片识别")

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    # dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # 7. 存储中间图片
    # cv2.imwrite(pic_path + "/binary.png", binary)
    # cv2.imwrite(pic_path + "/dilation.png", dilation)

    return dilation


def findTextRegion(org, img):
    regions = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxArea = 0
    maxContour = 0


    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(org, (x, y), (x + w, y + h), (255, 255, 0), 2)
        region = {}
        region['x'] = x
        region['y'] = y
        region['w'] = w
        region['h'] = h
        regions.append(region)

    return regions

def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(pic_path + "/gray.jpg", gray)
    # logger.info("已生成灰度图【%s】", pic_path + "/gray.jpg")
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    regions = findTextRegion(gray, dilation)
    wordInfos = []
    imgInfos = []
    for reg in regions:
        x = reg['x']
        y = reg['y']
        w = reg['w']
        h = reg['h']
        cropImg = gray[y:y + h, x:x + w]
        imgInfos.append(cropImg)
        # cv2.imwrite(pic_path + "/" + text +".png", cropImg)
        text = ''
        word_info = getInfo(x,y,w,h,text)
        wordInfos.append(word_info)
    text_list = httpUtil.imageArrayToTextList(imgInfos)
    for index in range(len(wordInfos)):
        if text_list[index]:
            logger.info("备注为【%s】,坐标为【%s】", text_list[index]['word'], wordInfos[index]['pos'])
            wordInfos[index]['word'] = text_list[index]['word']
    cv2.waitKey(0)
    return wordInfos

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

def gjj_start(img):
    wordInfos = detect(img)
    content = ""
    for word in wordInfos:
        content += word['word']
        content += " "
    height = img.shape[0]
    width = img.shape[1]
    result = {}
    result['content'] = content
    result['height'] = height
    result['orgHeight'] = height
    result['width'] = width
    result['orgWidth'] = width
    result['prism_version'] = "1.0"
    result['prism_wnum'] = len(wordInfos)
    result['prism_wordsInfo'] = wordInfos
    return result

if __name__ == '__main__':
    init_logger()
    # 读取文件
    img_name = "20196155"
    imagePath = "../data/gjj/" + img_name + ".png"
    logger.info("图片【%s】识别开始",imagePath)
    pic_path = pic_path + img_name
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    img = cv2.imread(imagePath)
    result = gjj_start(img)
    label_file = open(pic_path + "/gjj.txt", "w")
    label_file.write(json.dumps(result,ensure_ascii=False,indent=4))
    logger.info("识别完成，已生成【%s】",pic_path + "/gjj.txt")