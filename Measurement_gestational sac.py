import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 用于给图片添加中文字符
def ImgText_CN(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否为OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc', textSize, encoding="utf-8")  ##中文字体
    draw.text((left, top), text, textColor, font=fontText)  # 写文字
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 实现图片反色功能
def PointInvert(img):
    height, width = img.shape  # 获取图片尺寸
    for i in range(height):
        for j in range(width):
            pi = img[i, j]
            img[i, j] = 255 - pi
    return img


img = cv2.imread("gongjian1.bmp", 0)  # 加载彩色图
img1 = cv2.imread("gongjian1.bmp", 1)  # 加载灰度图

recimg = img[80:230, 90:230]  # 截取需要的部分
img2 = img1[80:230, 90:230]  # 截取需要的部分
ret, th = cv2.threshold(recimg, 90, 255, cv2.THRESH_BINARY)  # 阈值操作二值化

# canny边缘检测
edges = cv2.Canny(th, 30, 70)
res = PointInvert(edges)  # 颜色反转
# 显示图片
cv2.imshow('original', th)  # 显示二值化后的图，主题为白色，背景为黑色 更加容易找出轮廓
key = cv2.waitKey(0)
if key == 27:  # 按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 得到轮廓

cnt = contours[0]  # 取出轮廓

x, y, w, h = cv2.boundingRect(cnt)  # 用一个矩形将轮廓包围

img_gray = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)  # 将灰度转化为彩色图片方便画图

cv2.line(img_gray, (x, y), (x + w, y), (0, 0, 255), 2, 5)  # 上边缘

cv2.line(img_gray, (x, y + h), (x + w, y + h), (0, 0, 255), 2, 5)  # 下边缘
img1[80:230, 90:230] = img_gray  # 用带有上下轮廓的图替换掉原图的对应部分

res1 = ImgText_CN(img1, '宽度%d' % h, 25, 25, textColor=(0, 255, 0), textSize=30)  # 绘制文字
# 显示图片
cv2.imshow('original', res1)
key = cv2.waitKey(0)
if key == 27:  # 按esc键时，关闭所有窗口
    print(key)
    cv2.destroyAllWindows()