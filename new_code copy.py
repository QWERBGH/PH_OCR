#!/usr/bin/env python3
# -*- coding:utf-8 -*-

''

__author__ = 'Jessen Yu'

import os
import csv
import shutil
import codecs
import cv2
import re
import glob
import numpy as np
import difflib
import time
from utils.crnn_lib import CRNNLib
import pytesseract

zh_pattern = re.compile("[\u4e00-\u9fa5]+")
price_regex = re.compile("[^0-9.]")
recognizer = CRNNLib(gpu=True)


def read_dict():
    tmp_name_dict = set()
    with open("name_dictionary.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            new_line = line.strip()
            tmp_name_dict.add(new_line)

    return list(tmp_name_dict)


name_dict = read_dict()


def check_name(check_str):
    temp_result = difflib.get_close_matches(check_str, name_dict, n=1, cutoff=0.5)

    if len(temp_result) == 0:
        return check_str
    else:
        return temp_result[0]


def contain_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)

    return match


def get_modev(values, thrd):
    if len(values) == 0:
        return 0

    values = np.array(values)
    count = 0
    idx = 0
    for item in values:
        if count < len(np.where(np.abs(values - item) < thrd)[0]):
            count = len(np.where(np.abs(values - item) < thrd)[0])
            idx = item

    avg = 0
    num = 0
    for i in range(0, len(values)):
        if np.abs(values[i] - idx) < thrd:
            avg += values[i]
            num += 1

    avg = avg/num

    return avg


def col_project(image, end_height):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
    binary_img = cv2.bitwise_not(binary_img)

    kernel = np.uint8(np.ones((21, 21)))
    dilated_img = cv2.dilate(binary_img, kernel)

    # cut_imgp = dilated_img[0: end_height, :]
    # sumx = np.sum(cut_imgp, axis=1)
    # sumx = sumx//255

    # cv2.namedWindow("image", 0)
    # cv2.imshow("image", cut_imgp)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/mmmm.jpg", cut_imgp)
    # edgep = []
    # p = cut_imgp.shape[0] - 1
    # q = 0
    # while p >= 0:
    #     q = 0
    #     while q < cut_imgp.shape[1]:
    #         if cut_imgp[p][q] == 255:
    #             edgep.append(p)
    #
    #             break
    #         q += 1
    #     p -= 1
    #
    # modeh = get_modev(edgep, 50)
    # cv2.namedWindow("image", 0)
    # cv2.imshow("image", dilated_img)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/mmmm.jpg", dilated_img)
    cv2.normalize(dilated_img, dilated_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cut_img = dilated_img[0: end_height, :]
    col_pro_val = np.sum(cut_img, axis=0)

    ratio = 0
    if end_height/image.shape[0] > 0.8:
        ratio = 0.7
    elif end_height/image.shape[0] > 0.45:
        ratio = 0.6
    else:
        ratio = 0.3

    threshold = ratio * end_height
    # threshold = ratio * modeh
    for j in range(len(col_pro_val)):
        if col_pro_val[j] < threshold:
            col_pro_val[j] = 0

    column_list = []
    start_t = 0

    for j in range(len(col_pro_val) - 3):
        if col_pro_val[0] and j == 0:
            start_t = 0
        if col_pro_val[j] == 0 and col_pro_val[j + 1] > 0:
            start_t = j
        if col_pro_val[j] > 0 and col_pro_val[j + 1] == 0 and col_pro_val[j + 2] == 0\
                and col_pro_val[j + 3] == 0:
            column_list.append([start_t, j + 1])

    if col_pro_val[len(col_pro_val) - 1]:
        column_list.append([start_t, len(col_pro_val) - 1])

    return column_list, dilated_img


def get_col_list(line_img, column_list, dilate_line_img):
    interval = 5
    line_col_list = []
    height_v, width_v, c = line_img.shape
    for col in column_list:
        start_point, end_point = col
        col_pro_val = np.sum(dilate_line_img, axis=0)

        point = start_point
        start_line = 0
        while point >= 0:
            if col_pro_val[point] < 0.4 * height_v:
                q = point - 1
                inc_board = 1
                while q >= 0:
                    if col_pro_val[q] < 0.4 * height_v:
                        inc_board += 1
                        if inc_board > interval:
                            break
                    else:
                        break
                    q = q - 1

                if inc_board > interval:
                    start_line = point
                    break
            point -= 1

        end_line = width_v - 1
        point = end_point
        while point < width_v:
            if col_pro_val[point] < 0.4 * height_v:
                q = point + 1
                inc_board = 1
                while q < width_v:
                    if col_pro_val[q] < 0.4 * height_v:
                        inc_board = inc_board + 1
                        if inc_board > interval:
                            break
                    else:
                        break
                    q = q + 1

                if inc_board > interval:
                    end_line = point
                    break
            point += 1

        col_img = line_img[:, start_line:end_line]
        gray_image = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
        binary_img = cv2.bitwise_not(binary_img)
        col_height, col_width = binary_img.shape
        pos = col_height // 2
        sum_v = 0
        for col_index in range(col_width):
            sum_v += binary_img[pos][col_index]

        if sum_v == 0:
            pass
        else:
            line_col_list.append([start_line, end_line])

        line_index = 1
        while line_index < len(line_col_list):
            if line_col_list[line_index][0] - line_col_list[line_index - 1][1] > 25:
                col_img = line_img[:, line_col_list[line_index][0] - 25: line_col_list[line_index][0]]
                gray_image = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
                _, binary_img = cv2.threshold(gray_image, 230, 255, cv2.THRESH_BINARY)
                binary_img = cv2.bitwise_not(binary_img)
                col_height, col_width = binary_img.shape
                pos = col_height // 2
                sum_v = 0
                for col_index in range(col_width):
                    sum_v += binary_img[pos][col_index]

                if sum_v:
                    line_col_list[line_index][0] = line_col_list[line_index][0] - 25
            line_index = line_index + 1

    return line_col_list


def rotate_image(image):
    gray_img_v = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height_t, width_t = gray_img_v.shape
    # 直线检测
    lsd = cv2.createLineSegmentDetector(0)
    d_lines = lsd.detect(gray_img_v)

    # 选取符合要求的直线进行倾斜矫正
    lines = []
    for d_line in d_lines[0]:
        x0 = int(round(d_line[0][0]))
        y0 = int(round(d_line[0][1]))
        x1 = int(round(d_line[0][2]))
        y1 = int(round(d_line[0][3]))

        # cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        length = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        if length > 0.5 * width_t and abs(y1 - y0) < 0.2 * height_t:
            start_x = min(x0, x1)
            end_x = max(x0, x1)
            if x0 == start_x:
                start_y = y0
                end_y = y1
            else:
                start_y = y1
                end_y = y0

            lines.append([start_x, start_y, end_x, end_y])

    # cv2.namedWindow("src", 0)
    # cv2.imshow("src", image)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/yyy.jpg", image)

    avg_angle = 0
    line_nums = 1 if len(lines) == 1 else len(lines) - 1

    for line_index in range(line_nums):
        x0, y0, x1, y1 = lines[line_index]
        angles = (y1 - y0) / (x1 - x0) * 180 / np.pi
        avg_angle = avg_angle + angles
    if len(lines):
        avg_angle = avg_angle / len(lines)
    else:
        avg_angle = 0

    mask = np.zeros(shape=(height_t, width_t, 3), dtype="uint8")
    cv2.rectangle(mask, (0, 0), (width_t - 1, height_t - 1), (0, 0, 0), -1)
    for line in lines:
        x0, y0, x1, y1 = line
        if x1 - x0 > 0.1 * width_t:
            y0 += 2
            y1 += 2
            cv2.line(mask, (x0, y0), (x1, y1), (255, 255, 255), 8)
    image = np.maximum(image, mask)

    if len(lines):
        line = lines[len(lines) // 2]
        x0 = line[0]
        y0 = line[1]
        rotate_matrix = cv2.getRotationMatrix2D((x0, y0), avg_angle, 1.0)
        rotated_img = cv2.warpAffine(image, rotate_matrix, (width_t, height_t), borderValue=(255, 255, 255))

        return rotated_img
    else:
        return image


def get_mode(values):
    if len(values) == 0:
        return 0

    values = np.array(values)
    count = 0
    idx = 0
    for item in values:
        if count < len(np.where(np.abs(values - item) < 4)[0]):
            count = len(np.where(np.abs(values - item) < 4)[0])
            idx = item

    return idx


def process(imgp, colInfo, shotname):
    start_time = time.time()
    img_copy = imgp.copy()
    if img_copy.shape[0] > 1700 or img_copy.shape[1] > 1200:
        imgresize = cv2.resize(img_copy, (1200, 1700), interpolation=cv2.INTER_LINEAR)
        img = imgresize.copy()
    else:
        img = imgp.copy()

    coord =[]
    items = []
    result = {}

    colNum, itemN, price_pos, amount_pos, money_pos, type = colInfo
    # 打开 csv 文件
    csv_file_path = "/Users/caicloud/Desktop/result_SB/result.csv"
    csv_file = codecs.open(csv_file_path, "wb", "utf_8_sig")
    writer = csv.writer(csv_file)

    # gray_img = img[:, :, 2]
    rotate_img = rotate_image(img)
    rotate_imgp = rotate_img.copy()
    gray = rotate_img[:, :, 2]
    height, width = gray.shape

    # mser = cv2.MSER_create(_min_area=100)
    mser = cv2.MSER_create()
    regions, boxes = mser.detectRegions(gray)

    new_boxes_T = []
    new_boxes_B = []
    for box in boxes:
        x, y, w, h = box
        if w < 2 * h and h < 2 * w and w < 40 and h < 40:
            new_boxes_T.append([x, y, x + w, y + h])
            # new_boxes_T.append([x, y, w, h])
            new_boxes_B.append([x, y, x + w, y + h])

    new_boxes_T.sort(key=lambda sy: sy[1])
    new_boxes_B.sort(key=lambda sy: sy[3])

    lineD = 10

    new_row = []
    start = 0
    i = 0
    end = 0
    while i < len(new_boxes_T) - 3:
        if new_boxes_T[i + 1][1] - new_boxes_T[i][1] >= lineD and new_boxes_T[i + 2][1] - new_boxes_T[i][1] >= lineD \
                and new_boxes_T[i + 3][1] - new_boxes_T[i][1] >= lineD:
            end = i
            new_row.append([start, end])
            start = i + 1
        i += 1
    new_row.append([end, len(new_boxes_T) - 1])

    rect_list = []
    bottomVal = []
    topVal = []
    inc_boarder = 5
    for row_index in new_row:
        if row_index[0] == row_index[1]:
            continue
        left = 2
        right = width - 2
        # 增加对上下边界的判断,求众数
        num = row_index[1] - row_index[0]
        if num < 5:
            continue

        topVal = [x[1] for x in new_boxes_T[row_index[0]:row_index[1]]]
        bottomVal = [x[3] for x in new_boxes_B[row_index[0]:row_index[1]]]

        posT = get_mode(topVal)
        newTopVal = [x for x in topVal if abs(x - posT) < 4]
        newTopVal.sort()
        top = max(0, newTopVal[0] - inc_boarder)

        posB = get_mode(bottomVal)
        newBottomVal = [x for x in bottomVal if abs(x - posB) < 4]
        newBottomVal.sort()
        bottom = newBottomVal[len(newBottomVal) - 1]

        rect_list.append([left, top, right, bottom, len(newBottomVal)])

    line_h = []
    for line in rect_list:
        left, top, right, bottom, nump = line
        line_h.append(bottom - top)

    lineh = get_mode(line_h)

    gray_rotate = cv2.cvtColor(rotate_img, cv2.COLOR_BGR2GRAY)
    _, binaryImg = cv2.threshold(gray_rotate, 80, 255, cv2.THRESH_OTSU)
    binaryImgN = cv2.bitwise_not(binaryImg)
    sumy = np.sum(binaryImgN, axis=1)
    sumy = sumy//255

    # cv2.namedWindow("img", 0)
    # cv2.imshow("img", binaryImg)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/ppp.jpg", binaryImgN)

    hei = 100
    numK = 0
    flag = 0
    end_hp = height
    for pl in range(0, len(sumy)):
        if sumy[pl] < 10:
            ql = pl
            numK = 0
            while ql < len(sumy):
                if sumy[ql] < 10:
                    numK += 1
                else:
                    break
                ql += 1

            if numK > hei:
                flag = 1
                end_hp = pl
                break

    line_pos = len(rect_list) - 1
    pos_h = len(rect_list) - 1
    while line_pos > 0:
        if abs(rect_list[line_pos][3] - end_hp) < 5:
            pos_h = line_pos

        line_pos -= 1

    # diffVal = []
    # line_pos = len(rect_list) - 1
    # while line_pos > 0:
    #     diffV = abs(rect_list[line_pos][4] - rect_list[line_pos - 1][4])
    #     diffVal.append(diffV)
    #
    #     line_pos -= 1
    if flag == 1:
        thrd = rect_list[pos_h][4] - 10
    else:
        thrd = 65

    end_h = height
    line_pos = pos_h
    while line_pos > 0:
        ##20 改 65
        if rect_list[line_pos][4] > thrd and rect_list[line_pos][1] - rect_list[line_pos - 1][3] < lineh \
                and abs(rect_list[line_pos][3] - rect_list[line_pos][1] - lineh) < lineh \
                and abs(rect_list[line_pos - 1][3] - rect_list[line_pos - 1][1] - lineh) < lineh:
            end_h = rect_list[line_pos][3]
            break
        line_pos = line_pos - 1

    # print(rect_list)
    # print(end_h)

    line_p = 1
    inc = 0
    while line_p < len(rect_list) - 2:
        if (rect_list[line_p + 1][1] < rect_list[line_p][3] and rect_list[line_p + 1][3] > rect_list[line_p + 2][1]) \
                or (rect_list[line_p + 1][3] <= rect_list[line_p][3] + 3
                    and rect_list[line_p + 1][1] >= rect_list[line_p][1])\
                or rect_list[line_p + 1][3] - rect_list[line_p + 1][1] < lineh//2:
            rect_list.pop(line_p + 1)
            line_p = line_p - 1

        line_p = line_p + 1

    line_p = 0
    while line_p < len(rect_list) - 1:
        #if rect_list[line_p + 1][1] - rect_list[line_p][3] >= rect_list[line_p][3] - rect_list[line_p][1]:
        numv = (rect_list[line_p + 1][1] - rect_list[line_p][3]) // (lineh - 4)
        #if rect_list[line_p + 1][1] - rect_list[line_p][3] >= lineh - 4:
        if numv > 0:
            #numv = (rect_list[line_p + 1][1] - rect_list[line_p][3]) // (lineh - 4)
            pv = 0
            while pv < numv:
                rect_list.insert(line_p + pv + 1, [rect_list[line_p][0], rect_list[line_p][3] + pv * lineh,
                                              rect_list[line_p][2], rect_list[line_p][3] + (pv + 1) * lineh + 1, rect_list[line_p][4]])
                pv = pv + 1

            line_p = line_p - 1
        line_p = line_p + 1

    line_p = 3
    while line_p < len(rect_list) - 1:
        numv = (rect_list[line_p][3] - rect_list[line_p][1]) // (2 * lineh)

        if numv > 0:
            numv = (rect_list[line_p][3] - rect_list[line_p][1]) // (lineh) - 1
            rect_list[line_p][3] = rect_list[line_p][1] + lineh
            pv = 0
            while pv < numv:
                rect_list.insert(line_p + pv + 1, [rect_list[line_p][0], rect_list[line_p][3] + pv * lineh,
                                                   rect_list[line_p][2], rect_list[line_p][3] + (pv + 1) * lineh + 1,
                                                   rect_list[line_p][4]])
                pv = pv + 1

            line_p = line_p - 1
        line_p = line_p + 1


    # for box in rect_list:
    #     x, y, w, h, gh = box
    #     cv2.rectangle(rotate_img, (x, y), (w, h), (0, 255, 0), 2)
    #
    # cv2.namedWindow("img", 0)
    # cv2.imshow("img", rotate_img)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/tttt.jpg", rotate_img)

    # end_h = height
    # line_pos = len(rect_list) - 1
    # while line_pos >= 0:
    #     if rect_list[line_pos][4] > 10:
    #         end_h = rect_list[line_pos][3]
    #         break
    #     line_pos = line_pos - 1

    # line_h = []
    # for line in rect_list:
    #     left, top, right, bottom, nump = line
    #     line_h.append(bottom - top)
    #
    # lineh = get_mode(line_h)

    # end_h = height
    # line_pos = len(rect_list) - 1
    # while line_pos > 0:
    #     if rect_list[line_pos][4] > 20 and rect_list[line_pos - 1][3] - rect_list[line_pos][1] < lineh \
    #             and abs(rect_list[line_pos][3] - rect_list[line_pos][1] - lineh) < lineh \
    #             and abs(rect_list[line_pos - 1][3] - rect_list[line_pos - 1][1] - lineh) < lineh:
    #         end_h = rect_list[line_pos][3]
    #         break
    #     line_pos = line_pos - 1

    # print(rect_list)
    # print("end_h", end_h)
    end_time = time.time()
    print("time:", end_time - start_time)

    index = 0
    for rect in rect_list:
        index += 1
        line_image = rotate_img[rect[1]: rect[3], rect[0]: rect[2]]
        line_h, line_w = line_image.shape[:2]
        if line_h > line_w:
            continue
        result = recognizer.recognize(line_image)

        if "项目" in result or "名称" in result or "单价" in result:
            tmp_result = ("名称", "价格", "数量", "金额")
            writer.writerow(tmp_result)
            break

    if index >= len(rect_list):
        csv_file.close()

        result = {"code": -1, "message": "Failure", "new_img": "", "file_name": shotname, "data":[]}

        return result, rotate_img

    cut_img = rotate_img[rect_list[index][1]:, :]
    col_list, dilate_img = col_project(cut_img, end_h)

    if len(col_list) != colNum:
        ch_w = [x[2] for x in boxes]
        modew = get_mode(ch_w)

        remove_idx = []
        for c_i in range(0, len(col_list)):
            if col_list[c_i][1] - col_list[c_i][0] < 2 * modew:
                remove_idx.append(c_i)
            else:
                break

        # c_i = len(col_list) - 1
        # while c_i >= 0:
        #     if col_list[c_i][1] - col_list[c_i][0] < 2 * modew:
        #         remove_idx.append(c_i)
        #     else:
        #         break
        #     c_i -= 1

        if len(remove_idx) > 0:
            col_list_t = []
            for c_i in range(0, len(col_list)):
                if c_i not in remove_idx:
                    col_list_t.append(col_list[c_i])

            col_list = col_list_t.copy()

    # for line in col_list:
    #     cv2.line(rotate_img, (line[0], 0), (line[0], height), (255, 0, 0), 2)
    #     cv2.line(rotate_img, (line[1], 0), (line[1], height), (255, 0, 0), 2)
    #
    # cv2.namedWindow("pic", 0)
    # cv2.imshow("pic", rotate_img)
    # cv2.waitKey(0)
    # cv2.imwrite("/Users/caicloud/Desktop/result_SB/%s_c.jpg"%shotname, rotate_img)

    p = 0
    col_rect = []
    while p <= len(rect_list) - 1:
        if p >= index:
            line_image = rotate_img[rect_list[p][1]: rect_list[p][3], :]
            line_dilate_img = dilate_img[(rect_list[p][1] - rect_list[index - 1][3]):
                                         (rect_list[p][3] - rect_list[index - 1][3]), :]
            line_h, line_w = line_image.shape[:2]
            if line_h > line_w:
                p += 1
                continue

            # 获得该行的列图像列表，同时去除重复的
            col_line_list = get_col_list(line_image, col_list, line_dilate_img)
            new_col_line = []
            for c_i in range(0, len(col_line_list)):
                new_col_line.append([col_line_list[c_i][0], rect_list[p][1], col_line_list[c_i][1], rect_list[p][3]])

            col_rect.append(new_col_line)

        p += 1

    p = 0
    lineV = []
    idxHB = []
    while p < len(col_rect):
        col_line_list = col_rect[p]

        flag = 0
        c_i = 0
        idxB = []
        while c_i < len(col_line_list):
            x0, y0, x1, y1 = col_line_list[c_i]

            b_i = 0
            while b_i < len(col_line_list):
                if c_i != b_i:
                    x2, y2, x3, y3 = col_line_list[b_i]

                    minx = max(x0, x2)
                    maxx = min(x1, x3)

                    if minx < maxx:
                        minc = min(x0, x2)
                        maxc = max(x1, x3)

                        if maxx - minx > 0.5 * (maxc - minc):
                            flag = 1
                            idxB.append(c_i)
                            # lineV.append(p)
                            # c_i = len(col_line_list)
                            break

                b_i += 1
            c_i += 1

        if flag == 1:
            lineV.append(p)
            idxHB.append(idxB)

        p += 1

    pos_sd = []
    for p in range(0, len(col_list)):
        if p == 0:
            pos_sd.append([img.shape[1], 0])
        else:
            pos_sd.append([0, 0])

    p = 0
    numL = 0
    while p < len(col_rect):
        if p not in lineV:
            col_line_list = col_rect[p]
            if len(col_line_list) == len(col_list):
                for c_i in range(0, len(col_line_list)):
                    start, y0,  end, y1 = col_line_list[c_i]
                    posS, posE = pos_sd[c_i]

                    if c_i == 0:
                        posS = min(posS, start)
                        posE = max(posE, end)
                    else:
                        posS += start
                        posE += end

                    pos_sd[c_i][0] = posS
                    pos_sd[c_i][1] = posE
                numL += 1

        p += 1

    if numL != 0:
        for c_i in range(1, len(col_list)):
            pos_sd[c_i][0] = int(pos_sd[c_i][0]/numL)
            pos_sd[c_i][1] = int(pos_sd[c_i][1]/numL)

        p = 0
        pos_i = 0
        while p < len(col_rect):
            if p in lineV:
                for b_i in range(0, len(idxHB[pos_i])):
                    c_i = idxHB[pos_i][b_i]
                    col_rect[p][c_i][0] = pos_sd[c_i][0]
                    col_rect[p][c_i][2] = pos_sd[c_i][1]
                    # col_rect[p][c_i + 1][0] = pos_sd[c_i + 1][0]
                    # col_rect[p][c_i + 1][2] = pos_sd[c_i + 1][1]
                    # pass

                pos_i += 1

            p += 1

    
    # p = 0
    # while p < len(col_rect):
    #     col_line_list = col_rect[p]
    #
    #     for c_i in range(0, len(col_line_list)):
    #         x0, y0, x1, y1 = col_line_list[c_i]
    #         cv2.rectangle(rotate_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
    #
    #     p += 1
    #
    # cv2.namedWindow("image", 0)
    # cv2.imshow("image", rotate_img)
    # cv2.waitKey(0)
    # cv2.imwrite('/Users/caicloud/Desktop/result_SB/UUUUUU.jpg', rotate_img)

    p = 0
    while p < len(col_rect):
        col_line_list = col_rect[p]

        if len(col_line_list) == 0:
            p += 1
            continue

        start_pos_x = col_line_list[0][0]
        end_pos_x = col_line_list[len(col_line_list) - 1][2]
        start_pos_y = col_line_list[0][1]
        end_pos_y = col_line_list[0][3]

        # col_line_listp = col_line_list[:]

        #添加中间列切结果显示
        # for col_line in col_line_list:
        #     start, starty, end, endy = col_line
        #     # cv2.rectangle(rotate_imgp, (start, rect_list[p][1]), (end, rect_list[p][3]), (255, 0, 0), 1)
        #     cv2.rectangle(rotate_imgp, (start, starty), (end, endy), (255, 0, 0), 1)

        # b_set = set(tuple(x) for x in col_line_list)
        # b_set = [list(x) for x in b_set]
        # b_set.sort(key=lambda x: col_line_list.index(x))
        # col_line_list = b_set
        tmp_result = ()

        # 检查列向量长度
        if len(col_line_list) < 4:
            for col_line in col_line_list:
                start, starty, end, endy = col_line
                pic_img = rotate_img[starty: endy, start: end]
                result = recognizer.recognize(pic_img)
                tmp_result += (result,)
            writer.writerow(tmp_result)
            p += 1
            continue

        # 跳至并记录名称列
        endpv = 0
        real_index = 0
        while real_index < len(col_line_list):
            col_line = col_line_list[real_index]
            start, starty, end, endy = col_line
            pic_img = rotate_img[starty: endy, start: end]
            result = recognizer.recognize(pic_img)

            if contain_zh(result):
                pic_img = rotate_img[starty: endy, start: end]
                result = recognizer.recognize(pic_img)
                result = check_name(result)
                tmp_result += (result,)
                real_index += 1
                endpv = end
                break
            real_index += 1

        # 过滤剩下的数字列
        col_line_list = col_line_list[real_index:]

        flagV = 1
        if type == -1:
            if len(col_line_list) > 0:
                start, starty, end, endy = col_line_list[0]
                if start - endpv < 50:
                    flagV = 2
                else:
                    flagV = 1
        else:
            flagV = 1

        # recog_idx = [price_pos - 1, amount_pos - 1, money_pos - 1]
        # for c_i in range(0, len(col_line_list)):
        #     if c_i in recog_idx:
        #         col_line = col_line_list[c_i]
        #         start, starty, end, endy = col_line
        #         pic_img = rotate_img[starty: endy, start: end]
        #         # result = recognizer.recognize(pic_img)
        #         grayp = cv2.cvtColor(pic_img, cv2.COLOR_BGR2GRAY)
        #         _, binaryImg = cv2.threshold(grayp, 80, 255, cv2.THRESH_OTSU)
        #         result = pytesseract.image_to_string(binaryImg, config='--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789.')
        #         tmp_result += (result,)

        #注释源代码
        if len(col_line_list) == 4:
            start, starty, end, endy = col_line_list[-1]
            pic_img = rotate_img[starty: endy, start: end]
            result = recognizer.recognize(pic_img)
            result = price_regex.sub('', result)
            if result == "1" or result == "" or result == " ":
                col_line_list = col_line_list[:-1]
            # else:
            #     col_line_list = col_line_list[1:]

        if len(col_line_list) >= 5:
            col_line_list = col_line_list[0:-1]

        for col_index, col_line in enumerate(col_line_list):
            start, starty, end, endy = col_line
            pic_img = rotate_img[starty: endy, start: end]
            result = recognizer.recognize(pic_img)
            result = price_regex.sub('', result)
            if result.startswith("."):
                result = result[1:]
            if col_index == 0 and "." not in result:
                result = result[:-4] + "." + result[-4:]
            if col_index == 2 and "." not in result and len(result) > 1:
                result = result[:-2] + "." + result[-2:]
            if result == "." and (col_index == 0 or col_index == 1 or col_index == 2):
                result = ""
            tmp_result += (result,)

        # 为演示修改，添加小于8
        if len(tmp_result) >= 4 and len(tmp_result) < 8:
            if tmp_result[1].count(".") > 1:
                new_num = tmp_result[1].replace(".", "")
                tmp = list(tmp_result)
                tmp[1] = new_num[:-4] + "." + new_num[-4:]
                tmp_result = tuple(tmp)
            if tmp_result[2].count(".") > 1:
                new_num = tmp_result[2].replace(".", "")
                tmp = list(tmp_result)
                tmp[2] = new_num[:-1] + "." + new_num[-1:]
                tmp_result = tuple(tmp)
            if tmp_result[3].count(".") > 1:
                new_num = tmp_result[3].replace(".", "")
                tmp = list(tmp_result)
                tmp[3] = new_num[:-2] + "." + new_num[-2:]
                tmp_result = tuple(tmp)
            if (tmp_result[2] == "") \
                    and tmp_result[1] != "" \
                    and tmp_result[3] != "" \
                    and float(tmp_result[1]) != 0:
                num = str(int(float(tmp_result[3]) / float(tmp_result[1])))
                tmp_result = (tmp_result[0], tmp_result[1], num, tmp_result[3])
            if tmp_result[-1] == "0.00":
                # print(tmp_result)
                # if tmp_result[1].count(' ') >= 1:
                #     new_num = tmp_result[1].replace(' ', '')
                #     tmp = list(tmp_result)
                #     tmp[3] = new_num
                #     tmp_result = tuple(tmp)
                #     print(tmp_result)
                if tmp_result[2] == '':
                    new_num = '0'
                    tmp = list(tmp_result)
                    tmp[2] = new_num
                    tmp_result = tuple(tmp)
                    # print(tmp_result)

                if tmp_result[1] == '':
                    new_num = '0'
                    tmp = list(tmp_result)
                    tmp[1] = new_num
                    tmp_result = tuple(tmp)
                    # print(tmp_result)

                # print(tmp_result[1], tmp_result[2])
                # print('\n')
                num = str(float(tmp_result[1]) * float(tmp_result[2]))
                tmp_result = (tmp_result[0], tmp_result[1], tmp_result[2], num)

        coord = [start_pos_x, start_pos_y, end_pos_x - start_pos_x + 1, end_pos_y - start_pos_y + 1]

        flagp = 0
        numH = 0
        for i in range(1, len(tmp_result)):
            if tmp_result[i] =='':
                numH += 1

        if numH == len(tmp_result) - 1 or len(tmp_result) == 0 or len(tmp_result) < 4:
            flagp = 1

        if flagp == 1:
            pass
        else:
            if flagV == 1:
                if tmp_result[3].count('.') > 1:
                    str_pos = tmp_result[3].find('.')
                    tmp = list(tmp_result)
                    tmp[3] = tmp[3][:str_pos] + tmp[3][str_pos + 1:]
                    tmp_result = tuple(tmp)

                # if tmp_result[3] != '' and float(tmp_result[3]) > 10000:
                #     tmp = list(tmp_result)
                #     tmp[3] = str(float(tmp_result[1]) * float(tmp_result[2]))
                #     tmp_result = tuple(tmp)

                    # tmp_result[3] = str(float(tmp_result[1]) * float(tmp_result[2]))

                it = {"idx": p, "item": tmp_result[0], "itemcoord": coord,
                         "unit_price": tmp_result[1], "quantity": tmp_result[2], "amount": tmp_result[3]}

                items.append(it)
            elif len(tmp_result) >= 5:
                if tmp_result[4].count('.') > 1:
                    str_pos = tmp_result[4].find('.')
                    tmp = list(tmp_result)
                    tmp[4] = tmp[4][:str_pos] + tmp[4][str_pos + 1:]
                    tmp_result = tuple(tmp)

                # if tmp_result[4] != '' and float(tmp_result[4]) > 10000:
                #     tmp = list(tmp_result)
                #     tmp[4] = str(float(tmp_result[2]) * float(tmp_result[3]))
                #     tmp_result = tuple(tmp)
                    # tmp_result[4] = str(float(tmp_result[2]) * float(tmp_result[3]))

                it = {"idx": p, "item": tmp_result[0], "itemcoord": coord,
                      "unit_price": tmp_result[2], "quantity": tmp_result[3], "amount": tmp_result[4]}

                items.append(it)

            # it = {"idx": p, "item": tmp_result[0], "itemcoord": coord,
            #       "unit_price": tmp_result[1], "quantity": tmp_result[2], "amount": tmp_result[3]}
            #
            # items.append(it)
        # 新添加的板式
        # b_set = set(tuple(x) for x in col_line_listp)
        # b_set = [list(x) for x in b_set]
        # b_set.sort(key=lambda x: col_line_listp.index(x))
        # col_line_listp = b_set
        #
        # i = 0
        # while i < len(col_line_listp) - 1:
        #     startv = max(col_line_listp[i][0], col_line_listp[i + 1][0])
        #     endv = min(col_line_listp[i][1], col_line_listp[i + 1][1])
        #     if startv < endv:
        #         col_line_listp.pop(i + 1)
        #
        #         j = 0
        #         while j < len(col_list) - 1:
        #             if col_line_listp[i][1] > col_list[j + 1][0] and col_line_listp[i][0] <= col_list[j][1]:
        #                 #col_line_listp[i][0] = startv
        #                 temp = col_line_listp[i][1]
        #                 col_line_listp[i][1] = col_list[j + 1][0]
        #
        #                 col_line_listp.insert(i + 1, [col_list[j + 1][0], temp])
        #                 break
        #             j = j + 1
        #
        #         i = i - 1
        #     i = i + 1
        #
        # # j = 0
        # # while j < len(col_line_listp):
        # #     startv, endv = col_line_listp[j]
        # #
        # #     i = 0
        # #     while i < len(col_list) - 1:
        # #         if endv > col_list[i + 1][0]:
        # #             col_line_listp[j][0] = startv
        # #             col_line_listp[j][1] = col_list[i][1]
        # #             col_line_listp.insert(j + 1, [col_list[i + 1][0], col_list[i + 1][1]])
        # #             j = j - 1
        # #             break
        # #         i = i + 1
        # #     j = j + 1
        #
        # # cv2.namedWindow('str', 0)
        # # cv2.imshow('str', line_image)
        # # cv2.waitKey(0)
        #
        # if len(col_line_listp) > 11:
        #     tmp_result = []
        #     col_line = col_line_listp[0]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     if contain_zh(result):
        #         pic_img = line_image[:, start: end]
        #         result = recognizer.recognize(pic_img)
        #         result = check_name(result)
        #         #tmp_result += (result,)
        #         tmp_result.append(result)
        #
        #
        #     col_line = col_line_listp[1]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)
        #
        #     col_line = col_line_listp[2]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)
        #
        #     col_line = col_line_listp[4]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)
        #
        #     col_line = col_line_listp[7]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     if contain_zh(result):
        #         pic_img = line_image[:, start: end]
        #         result = recognizer.recognize(pic_img)
        #         result = check_name(result)
        #         #tmp_result += (result,)
        #         tmp_result.append(result)
        #
        #     col_line = col_line_listp[8]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)
        #
        #     col_line = col_line_listp[9]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)
        #
        #     col_line = col_line_listp[11]
        #     start, end = col_line
        #     pic_img = line_image[:, start: end]
        #     result = recognizer.recognize(pic_img)
        #     #tmp_result += (result,)
        #     tmp_result.append(result)

            # tmp_result = (tmp_result[0], tmp_result[1], tmp_result[2], tmp_result[4], tmp_result[7], tmp_result[8],
            #               tmp_result[9], tmp_result[11])

        # print(tmp_result)

        writer.writerow(tmp_result)
        p = p + 1

    data = {"xm": "", "zysj": "", "items": items}

    result = {"code": 0, "message": "ok", "new_img": "", "file_name": shotname, "data": data}

    # cv2.imwrite("/Volumes/Macintosh HD/1.jpg", rotate_imgp)
    csv_file.close()

    return result, rotate_img


image_list = glob.glob("/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/复旦大学附属中山医院/徐玉华/*.jpg")
for image_path in image_list:
    image_path = "/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/[0831]浙江大学医学院附属第一医院/朱增珍/Image_00002.jpg"
    (filepath, tempfilename) = os.path.split(image_path)
    (shotname, extension) = os.path.splitext(tempfilename)
    print(image_path)

    imgp = cv2.imread(image_path)
    # print(imgp.shape[0], imgp.shape[1])
    # img = cv2.imread("/Users/caicloud/PROJECT/Social Security/pic/pic/浙江大学医学院/浙江大学医学院附属第一医院/Image_00003.jpg")
    # img_copy = imgp.copy()
    # if img_copy.shape[0] > 1700 or img_copy.shape[1] > 1200:
    #     imgresize = cv2.resize(img_copy, (1200, 1700), interpolation=cv2.INTER_LINEAR)
    #     img = imgresize.copy()
    # else:
    #     img = imgp.copy()

    #

    colInfo = [6, 0, 1, 2, 3]
    hostipalName = "浙江大学第一附属医院"
    if hostipalName == "复旦大学附属中山医院":
        colInfo.append(-1)
    else:
        colInfo.append(0)

    result, rotate_img = process(imgp, colInfo, shotname)
    # rotate_img = process(img, colInfo, shotname)
    #
    cv2.namedWindow("rotate", 0)
    cv2.imshow("rotate", rotate_img)
    cv2.waitKey(0)
    cv2.imwrite("/Users/caicloud/Desktop/result_SB/%s_r.jpg"%shotname, rotate_img)

    p = 0
