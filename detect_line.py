#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np
import os


def remove_lines(image, linesp, width=8):
    # 直线去除
    # for line in linesp:
    #     x0, y0, x1, y1 = line
    #     length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    #     if length > 0.95 * width_t or length > 0.1 * height_t:
    #         cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
    if image.ndim == 3:
        height_t, width_t, channel = image.shape
        mask = np.zeros(shape=(height_t, width_t, 3), dtype="uint8")
        cv2.rectangle(mask, (0, 0), (width_t - 1, height_t - 1), (0, 0, 0), -1)
        for line in linesp:
            x0, y0, x1, y1 = line
            # if x1 - x0 > 0.1 * width_t:
            y0 += 2
            y1 += 2
            cv2.line(mask, (x0, y0), (x1, y1), (255, 255, 255), width)

            x0 += 2
            x1 += 2
            cv2.line(mask, (x0, y0), (x1, y1), (255, 255, 255), width)
        image = np.maximum(image, mask)
        # cv2.imwrite("/Users/caicloud/PROJECT/电力OCR/质量较差的图像/1.jpg", image)
    elif image.ndim == 2:
        height_t, width_t= image.shape
        mask = np.zeros(shape=(height_t, width_t), dtype="uint8")
        cv2.rectangle(mask, (0, 0), (width_t - 1, height_t - 1), (0, 0, 0), -1)
        for line in linesp:
            x0, y0, x1, y1 = line
            # if x1 - x0 > 0.1 * width_t:
            y0 += 2
            y1 += 2
            cv2.line(mask, (x0, y0), (x1, y1), (255, 255, 255), 12)

            x0 += 2
            x1 += 2
            cv2.line(mask, (x0, y0), (x1, y1), (255, 255, 255), 12)
        imagep = cv2.bitwise_not(image, image)
        imaget = np.maximum(imagep, mask)
        image = cv2.bitwise_not(imaget, imaget)
        # cv2.imwrite("/Users/caicloud/PROJECT/电力OCR/质量较差的图像/1.jpg", image)

    return image


def detect_line(image, minlen=30):
    if image.ndim == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    height_t, width_t = gray_img.shape
    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(gray_img)

    if dlines[0] is None:
        return []
    # 直线链接
    idx = 0
    p = 0
    PI = 3.1415926
    linesp = []
    lindex = []
    # linesv = []
    i = j = 0
    while j < len(dlines[0]):
        if j in lindex:
            j = j + 1
            continue

        linet = dlines[0][j]
        x0 = int(round(linet[0][0]))
        y0 = int(round(linet[0][1]))
        x1 = int(round(linet[0][2]))
        y1 = int(round(linet[0][3]))

        # length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        # if length > 50:
        #     linesv.append([x0, y0, x1, y1])

        # x0 = int(round(dline[0][0]))
        # y0 = int(round(dline[0][1]))
        # x1 = int(round(dline[0][2]))
        # y1 = int(round(dline[0][3]))

        # cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
        # continue

        start_x = min(x0, x1)
        end_x = max(x0, x1)
        start_y = min(y0, y1)
        end_y = max(y0, y1)

        line_piece = []
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

        if length > 0.8 * width_t or length > 0.07 * height_t:
            linesp.append([x0, y0, x1, y1])
            p = p + 1
            idx = idx + 1
            j = j + 1
            continue

        idx = idx + 1
        if length > minlen:
            line_piece.append([start_x, start_y, end_x, end_y])
            lindex.append(j)
            if x1 != x0:
                angle = (end_y - start_y) / (end_x - start_x) * 180 / PI
            elif y0 == start_y:
                angle = 90
            else:
                angle = 270

            while i < len(dlines[0]):
                if i in lindex:
                    i = i + 1
                    continue

                line = dlines[0][i]
                x2 = int(round(line[0][0]))
                y2 = int(round(line[0][1]))
                x3 = int(round(line[0][2]))
                y3 = int(round(line[0][3]))

                start_xt = min(x2, x3)
                end_xt = max(x2, x3)
                start_yt = min(y2, y3)
                end_yt = max(y2, y3)
                if x2 != x3:
                    angle1 = (end_yt - start_yt) / (end_xt - start_xt) * 180 / PI
                elif y0 == start_y:
                    angle1 = 90
                else:
                    angle1 = 270
                length = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5

                i = i + 1
                if length < minlen - 10 or length > 0.8 * width_t or length > 0.07 * height_t:
                    i = i + 1
                    continue

                # 加上有线段有重叠情况
                #if x0 != x2 and y0 != y2 and x1 != x3 and y1 != y3:
                if i != j:
                    if (start_xt - end_x >= 0 and start_xt - end_x < 10 \
                            and start_yt - end_y >= 0 and start_yt - end_y < 10 and abs(angle - angle1) < 5):
                        line_piece.append([start_xt, start_yt, end_xt, end_yt])
                        lindex.append(i)
                        end_x = end_xt
                        end_y = end_yt
                        angle = angle1
                        i = 0

        if len(line_piece) > 1:
            line_piece.sort(key=lambda sx: sx[0])

            start_x = line_piece[0][0]
            start_y = line_piece[0][1]
            end_x = line_piece[len(line_piece) - 1][2]
            end_y = line_piece[len(line_piece) - 1][3]
            linesp.append([start_x, start_y, end_x, end_y])

        j = j + 1

    # return linesp, linesv
    return linesp


def hough_line_detect(image):
    # imgcpy = image.copy()
    # if imgcpy.shape[0] > 1700 or imgcpy.shape[1] > 1200:
    #     imgresize = cv2.resize(imgcpy, (1200, 1700), interpolation=cv2.INTER_LINEAR)

    # save_path = "/Users/caicloud/Desktop/result_SB/7_p.jpg"
    # cv2.imwrite(save_path, imgresize)

    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (3, 3), 0)
    edges = cv2.Canny(blurImg, 50, 150, apertureSize=3)
    # save_path = "/Users/caicloud/Desktop/result_SB/5_edges.jpg"
    # cv2.imwrite(save_path, edges)
    blurIm2 = cv2.GaussianBlur(edges, (3, 3), 0)
    _, binaryImg = cv2.threshold(blurIm2, 80, 255, cv2.THRESH_OTSU)
    # save_path = "/Users/caicloud/Desktop/result_SB/5_p.jpg"
    # cv2.imwrite(save_path, binaryImg)
    kernel = np.uint8(np.ones((5, 5)))
    dilateImg = cv2.dilate(binaryImg, kernel)
    dilateImgN = cv2.bitwise_not(dilateImg)
    # save_path = "/Users/caicloud/Desktop/result_SB/9_p.jpg"
    # cv2.imwrite(save_path, dilateImg)
    # _, binaryImg2 = cv2.threshold(dilateImg, 80, 255, cv2.THRESH_OTSU)
    # save_path = "/Users/caicloud/Desktop/result_SB/11_p.jpg"
    # cv2.imwrite(save_path, dilateImgN)

    # lines = cv2.HoughLines(edges, 1, np.pi/180, min(int(0.3 * grayImg.shape[0]), int(0.3 * grayImg.shape[1])))
    lines = cv2.HoughLinesP(dilateImg, 1, np.pi/180, 30, maxLineGap=20)

    linesp = []
    lines1 = lines[:, 0, :]
    # for rho, theta in lines1[:]:
    for line in lines1[:]:
        # a = np.cos(theta)
        # b = np.sin(theta)
        #
        # x0 = a * rho
        # y0 = b * rho
        # x1 = int(x0 + 1000 * (-b))
        # y1 = int(y0 + 1000 * (a))
        # x2 = int(x0 - 1000 * (-b))
        # y2 = int(y0 - 1000 * (a))

        x1, y1, x2, y2 = line
        #
        # linesp.append([x1, y1, x2, y2])

        # if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
        #     # pt1 = (int(rho / np.cos(theta)), 0)
        #     # # 该直线与最后一行的焦点
        #     # pt2 = (int((rho - grayImg.shape[0] * np.sin(theta)) / np.cos(theta)), grayImg.shape[0])
        #     # # 绘制一条白线
        #     # cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        #     pass
        # el
        # if 80 * np.pi/180 < theta <= 100 * np.pi/180:  # 水平直线
            # 该直线与第一列的交点
            # pt1 = (0, int(rho / np.sin(theta)))
            # # 该直线与最后一列的交点
            # pt2 = (grayImg.shape[1], int((rho - grayImg.shape[1] * np.cos(theta)) / np.sin(theta)))
            # # 绘制一条直线
            # cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > 0.4 * image.shape[1]:
            pt1 = (x1, y1)
            pt2 = (x2, y2)
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)

    cv2.namedWindow("image", 0)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    save_path = "/Users/caicloud/Desktop/result_SB/1_p.jpg"
    cv2.imwrite(save_path, image)

    return linesp


# nump = 1
if __name__ == '__main__':
    image_list = glob.glob("/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/浙江大学医学院附属第二医院/*.jpg")
    for image_path in image_list:
        image_path='/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/浙江大学医学院附属第二医院/Image_00016.jpg'
        (filepath, tempfilename) = os.path.split(image_path)
        (shotname, extension) = os.path.splitext(tempfilename)
        image = cv2.imread(image_path)
        # image = cv2.imread('/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/浙江大学医学院附属第二医院/Image_00016.jpg')
        print(image_path)

        # 直线检测1
        # linesp = detect_line(image)
        # processImg = remove_lines(image, linesp)
        # save_path = "/Users/caicloud/Desktop/result_SB/%s.jpg"%shotname
        # cv2.imwrite(save_path, processImg)

        # for line in linesv:
        #     x0, y0, x1, y1 = line
        #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 5)
        #
        # save_path = "/Users/caicloud/Desktop/result_SB/%s_p.jpg" % shotname
        # cv2.imwrite(save_path, image)

        # nump = nump + 1


        # 直线检测2
        linesp = hough_line_detect(image)
        # for line in linesp:
        #     x0, y0, x1, y1 = line
        #     cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        #
        # cv2.namedWindow("image", 0)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # save_path = "/Users/caicloud/Desktop/result_SB/%s_p.jpg" % shotname
        # cv2.imwrite(save_path, image)

