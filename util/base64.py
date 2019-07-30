
import base64,cv2,numpy as np,logging

def base64_to_cv2(base64_img):
    # 去掉可能传过来的“data:image/jpeg;base64,”HTML tag头部信息
    index = base64_img.find(",")
    if index != -1: base64_img = base64_img[index + 1:]
    imgString = base64.b64decode(base64_img)
    nparr = np.fromstring(imgString, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def cv2_to_base64(image):
    base64_str = cv2.imencode('.jpg', image)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str