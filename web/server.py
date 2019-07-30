#-*- coding:utf-8 -*-
from flask import Flask,jsonify,request,render_template
import base64,cv2,numpy as np,logging
from main import gjj_ocr
from util import base64
import  threading
import  time
import json
app = Flask(__name__,root_path="web")
person_img_num = 10
logger = logging.getLogger("WebServer")
lock = threading.Lock()

@app.route("/")
def index():
    return "哈哈"

# 公积金图片识别
@app.route('/gjj_ocr',methods=['POST'])
def gjj():
    base64_img = request.form.get('img')
    img = base64.base64_to_cv2(base64_img)
    if img is None:
        logger.error("图像解析失败")#有可能从字节数组解析成图片失败
        return "图像角度探测失败"

    logger.debug("从字节数组变成图像的shape:%r",img.shape)
    wordsInfo = gjj_ocr.gjj_start(img)
    return jsonify({'result': wordsInfo})

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

init_logger()

