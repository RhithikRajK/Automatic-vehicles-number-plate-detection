# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from os import listdir
import imutils
import glob

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
folder='D:\car_dir1'
image=Path(folder).glob('*.jpg')
print("Original Number Plate", "\t", "Predicted Number Plate", "\t", "Accuracy")
print("--------------------", "\t", "-----------------------", "\t", "--------")
for images in image:
    file_path = str(images)
    file_path1 = os.path.basename(images)
    NP_list = []
    predicted_NP = []
    number_plate = os.path.splitext(file_path1)[0]
    number_plate=str(number_plate)
    NP_list.append(number_plate)

    NP_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(NP_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)
    #thresh = cv2.threshold(NP_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
        print("No contour detected")
    else:
        detected = 1
    if detected == 1:
        cv2.drawContours(NP_img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(NP_img, NP_img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    predicted_res = pytesseract.image_to_string(Cropped, lang='eng', config='--psm 6')
    predicted_res=predicted_res.strip()
    predicted_res=str(predicted_res)

    fin_str=""
    for i in predicted_res:
        if i!=" ":
            fin_str+=i
    predicted_NP.append(fin_str)


    def estimate_predicted_accuracy(ori_list, pre_list):
        for ori_plate, pre_plate in zip(ori_list, pre_list):
            acc = "0 %"
            number_matches = 0
            if ori_plate == pre_plate:
                acc = "100 %"
            else:
                if len(ori_plate) == len(pre_plate):
                    for o, p in zip(ori_plate, pre_plate):
                        if o == p:
                            number_matches += 1
                    acc = str(round((number_matches / len(ori_plate)), 2) * 100)
                    acc += "%"
            print(ori_plate, "              ", pre_plate, "                  ", acc)
    estimate_predicted_accuracy(NP_list, predicted_NP)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
