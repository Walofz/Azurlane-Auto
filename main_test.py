import cv2
import numpy
import subprocess
import os
from imutils import contours, grab_contours
from adb import Adb
from multiprocessing.pool import ThreadPool

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_numbers(x, y, w, h, digits=5):
    text = []

    crop = screen[y: y + h, x: x + w]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method="left-to-right")[0]

    if len(cnts) > digits:
        return 0

    for c in cnts:
        scores = []

        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y: y + h, x: x + w]
        row, col = roi.shape[:2]

        width = round(abs((50 - col)) / 2) + 5
        height = round(abs((94 - row)) / 2) + 5
        resized = cv2.copyMakeBorder(
            roi,
            top=height,
            bottom=height,
            left=width,
            right=width,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        for x in range(0, 10):
            template = cv2.imread("assets/numbers/{}.png".format(x), 0)

            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        text.append(str(numpy.argmax(scores)))

    text = "".join(text)
    print(text)
    return int(text)


def import_image():
    global screen
    screen = None
    while screen is None:
        if Adb.legacy:
            screen = cv2.imdecode(
                numpy.frombuffer(
                    exec_out(r"screencap -p | sed s/\r\n/\n/"), dtype=numpy.uint8), 0,)
        else:
            screen = cv2.imdecode(numpy.frombuffer(
                exec_out("screencap -p"), dtype=numpy.uint8), 0)


def exec_out(args):
    # cmd = 'adb exec-out screencap -p'
    cmd = ["adb", "exec-out"] + args.split(" ")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return process.communicate()[0]


import_image()
read_numbers(970, 38, 101, 36)
