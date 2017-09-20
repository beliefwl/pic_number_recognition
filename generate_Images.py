# coding=utf-8
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


#训练集路径
image_path_train = "/home/tuixiang/dataset/pic_num_recognition/train/"
if not os.path.exists(image_path_train):
    os.makedirs(image_path_train)


#测试集路径
image_path_test = "/home/tuixiang/dataset/pic_num_recognition/test/"
if not os.path.exists(image_path_test):
    os.makedirs(image_path_test)


#字体
DEFAULT_FONTS = "/opt/pycharm-2017.2.3/jre64/lib/fonts/DroidSansMono.ttf"


#图片大小
WIDHT = 28
HEIGHT = 28


#训练 测试数据总量
train_total = 5000
test_total = 200


#随机生成图片显示的数字
def gene_num():
    return str(random.randint(0,9));


# 用来生成文件名前缀 最终文件名为 123456_9.jpg
def gene_filename():
    return str(int(random.random() * 1000000000));


# 用opencv 转为灰度
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def generate_image(num, dir_path):

    #保存文件名
    filename = gene_filename() + "-" + str(num) + ".jpg"
    path = dir_path + filename
    print(path)

    color = (0, 0, 0)
    background = (255, 255, 255)
    print("要存的数字是" + num)
    image = create_image_one_char(num, color, background)
    image = convert2gray(np.array(image))
    cv2.imwrite(path, image)

def create_image_one_char(c, color, background):
    font = ImageFont.truetype(DEFAULT_FONTS, 30)
    im = Image.new('RGBA', (WIDHT, HEIGHT), background)
    drawAvatar = ImageDraw.Draw(im)
    w, h = im.size
    drawAvatar.text((4, -3), c, fill=color, font=font)
    del drawAvatar
    # rotate
    im = im.crop(im.getbbox())
    im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)
    # warp
    dx = w * random.uniform(0.1, 0.4)
    dy = h * random.uniform(0.2, 0.5)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )
    im = im.resize((w2, h2))
    im = im.transform((WIDHT, HEIGHT), Image.QUAD, data)
    image = Image.new('RGB', (WIDHT, HEIGHT), background)
    image.paste(im, (0, 0), im)
    return image



def main():
    #训练集
    for x in range(0,train_total):
        generate_image(gene_num(), image_path_train)


    #测试集
    for x in range(0, test_total):
        generate_image(gene_num(), image_path_test)


if __name__ == '__main__':
    main()