import util.opencvUtil as opencvUtil
import matplotlib
matplotlib.use('TkAgg')
import cv2
import os
from skimage import io,draw,transform,color
import numpy as np
import json
import pytesseract
from util import httpUtil

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
    height = img.shape[0]
    width = img.shape[1]
    # 姓名
    name_img = img[int(height*0.11):int(height*0.24), int(width*0.18):int(width*0.4)]
    cv2.imwrite(pic_path + "/name.jpg", name_img)
    wordInfo['name'] = httpUtil.imageToString(name_img)
    logger.info("姓名为【%s】",wordInfo['name'])
    # 性别
    sex_img = img[int(height * 0.25):int(height * 0.35),int(width * 0.18):int(width * 0.25)]
    cv2.imwrite(pic_path + "/sex.jpg", sex_img)
    wordInfo['sex'] = httpUtil.imageToString(sex_img)
    logger.info("性别为【%s】",wordInfo['sex'])
    # 民族
    nation_img = img[int(height * 0.24):int(height * 0.34),int(width * 0.39):int(width * 0.44)]
    cv2.imwrite(pic_path + "/nation.jpg", nation_img)
    wordInfo['nation'] = httpUtil.imageToString(nation_img)
    logger.info("名族为【%s】",wordInfo['nation'])
    # 生日
    birthday_img = img[int(height * 0.35):int(height * 0.48), int(width * 0.18):int(width * 0.61)]
    cv2.imwrite(pic_path + "/birthday.jpg", birthday_img)
    wordInfo['birth'] = httpUtil.imageToString(birthday_img)
    logger.info("生日为【%s】",wordInfo['birth'])
    # 地址
    address_img_1 = img[int(height * 0.47):int(height * 0.58), int(width * 0.17):int(width * 0.63)]
    address_img_2 = img[int(height * 0.59):int(height * 0.68), int(width * 0.17):int(width * 0.63)]
    cv2.imwrite(pic_path + "/address1.jpg", address_img_1)
    cv2.imwrite(pic_path + "/address2.jpg", address_img_2)
    wordInfo['address'] = httpUtil.imageToString(address_img_1) + httpUtil.imageToString(address_img_2)
    logger.info("地址为【%s】",wordInfo['address'])
    # 身份证号
    idcard_img = img[int(height * 0.8):int(height * 0.91), int(width * 0.34):int(width * 0.93)]
    cv2.imwrite(pic_path + "/idcard.jpg", idcard_img)
    wordInfo['num'] = httpUtil.imageToString(idcard_img)
    logger.info("身份证号为【%s】",wordInfo['num'])
    return wordInfo


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(pic_path + "/gray.jpg", gray)
    logger.info("已生成灰度图【%s】", pic_path + "/gray.jpg")

    # 2. 高斯滤波
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # 3. 自适应二值化方法
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    cv2.imwrite(pic_path + "/binary.png", binary)
    # 腐蚀
    # erode = opencvUtil.erode(binary,5)
    # cv2.imwrite(pic_path + "/erode.png", erode)
    # 降噪
    # noise = opencvUtil.noise_remove_cv2(binary,2)
    # cv2.imwrite(pic_path + "/noise.png", noise)
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
            # opencv中对指定的点集进行多边形逼近的函数
            # arg1:输入的点集 arg2:指定的精度,也即是原始曲线与近似曲线之间的最大距离  arg3:若为true，则说明近似曲线是闭合的；反之，若为false，则断开
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果近似轮廓有四个顶点，那么就认为找到了
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
    # warped = opencvUtil.grayImg(warped)
    # warped = opencvUtil.binary(warped,120)
    cv2.imwrite(pic_path + "/warped.png", warped)

    # 7. 划分文字区域
    wordInfo = findTextRegion(warped)
    return wordInfo


# def four_point_transform(image, docCnt):
#     # 自定义
#     # x1,x2 = max(docCnt[0][0],docCnt[1][0]),min(docCnt[2][0],docCnt[3][0])
#     # y1,y2 = max(docCnt[0][1],docCnt[3][1]),min(docCnt[1][1],docCnt[2][1])
#     # cut_img = image[y1:y2, x1:x2]
#     # opencv
#     x, y, w, h = cv2.boundingRect(docCnt)
#     cut_img = image[y:y + h, x:x + w]
#     return cut_img

# 4点透射变换
def four_point_transform(image, docCnt):
    # 自定义
    # x1,x2 = max(docCnt[0][0],docCnt[1][0]),min(docCnt[2][0],docCnt[3][0])
    # y1,y2 = max(docCnt[0][1],docCnt[3][1]),min(docCnt[1][1],docCnt[2][1])
    # cut_img = image[y1:y2, x1:x2]
    # opencv
    # 原图
    src = np.array([[docCnt[0][0],docCnt[0][1]],[docCnt[3][0],docCnt[3][1]],[docCnt[1][0],docCnt[1][1]],[docCnt[2][0],docCnt[2][1]]],np.float32)
    # 高和宽
    h,w = image.shape[:2]
    # 目标图
    dst = np.array([[0,0],[w,0],[0,h],[w,h]],np.float32)
    P = cv2.getPerspectiveTransform(src, dst)  # 计算投影矩阵
    r = cv2.warpPerspective(img, P, (w, h), borderValue=125)
    return r


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
    wordInfo = detect(img)
    return wordInfo

if __name__ == '__main__':
    init_logger()
    # 读取文件binary
    img_name = "yh"
    imagePath = "../data/idcard/" + img_name + ".jpg"
    logger.info("图片【%s】识别开始",imagePath)
    pic_path = pic_path + img_name
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    img = cv2.imread(imagePath)
    wordInfo = detect(img)
    label_file = open(pic_path + "/idcard.txt", "w")
    label_file.write(json.dumps(wordInfo,ensure_ascii=False,indent=4))
    logger.info("识别完成，已生成【%s】",pic_path + "/idcard.txt")