from PIL import Image
import pytesseract

if __name__ == '__main__':
    # text = pytesseract.image_to_string(Image.open('../data/idcard_pic/yh.jpg'), lang='chi_sim')
    text = pytesseract.image_to_string(Image.open('../data/gjj/19469159.png'), lang='chi_sim')
    print(text)