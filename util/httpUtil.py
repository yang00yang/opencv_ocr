import json
from util import base64
import requests

deep_url = "http://127.0.0.1:8081/crnn"

def imageToString(img):
    base64_data = base64.cv2_to_base64(img)
    imgStr = base64_data.decode()
    content = []
    img = {"img": imgStr}
    content.append(img)
    content = json.dumps(content)
    r = requests.post(url=deep_url, data=content)
    c = json.loads(r.content)
    prism_wordsInfo = c["prism_wordsInfo"]
    return  prism_wordsInfo[0]["word"]


def imageArrayToTextList(imgs):
    content = []
    for img in imgs:
        base64_data = base64.cv2_to_base64(img)
        imgStr = base64_data.decode()
        img = {"img": imgStr}
        content.append(img)
    content = json.dumps(content)
    r = requests.post(url=deep_url, data=content)
    print("得到结果为" + str(r.content))
    c = json.loads(r.content.decode('utf-8'))
    prism_wordsInfo = c["prism_wordsInfo"]
    return prism_wordsInfo,c["sid"]
