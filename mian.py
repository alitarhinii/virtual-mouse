import cv2 as cv
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Load header images
path = 'header'
headerlist = os.listdir(path)
image_list = [cv.imread(f'{path}/{img}') for img in headerlist]
desired_width = 1280
scale_factor = desired_width / image_list[0].shape[1]
desired_height = int(image_list[0].shape[0] * scale_factor)
image_list = [cv.resize(img, (desired_width, desired_height)) for img in image_list]

header = image_list[0]

detector = HandDetector()
xp, yp = 0, 0
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
color = (255, 0, 0)
image_canvas = np.ones((720, 1280, 3), np.uint8)

while True:
    suc, img = cap.read()
    img = cv.flip(img, 1)
    if not suc:
        break

    hands, img = detector.findHands(img, draw=False)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]
            if len(lmList) != 0:
                x1, y1 = lmList[8][0], lmList[8][1]
                x2, y2 = lmList[12][0], lmList[12][1]
                fingers_list = detector.fingersUp(hand)

                if fingers_list[1] and fingers_list[2]:
                    cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), color, cv.FILLED)
                    xp, yp = 0, 0
                    if y1 < header.shape[0]:
                        if 250 < x1 < 270:
                            header, color = image_list[0], (255, 0, 0)
                        elif 550 < x1 < 600:
                            header, color = image_list[1], (0, 0, 255)
                        elif 850 < x1 < 900:
                            header, color = image_list[2], (0, 255, 0)
                        elif 1120 < x1 < 1150:
                            header, color = image_list[3], (0, 0, 0)

                if fingers_list[1] and not fingers_list[2]:
                    cv.circle(img, (x1, y1), 15, color, cv.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    thickness = 15 if color != (0, 0, 0) else 50
                    cv.line(img, (xp, yp), (x1, y1), color, thickness)
                    cv.line(image_canvas, (xp, yp), (x1, y1), color, thickness)
                    xp, yp = x1, y1

    mask = cv.threshold(cv.cvtColor(image_canvas, cv.COLOR_BGR2GRAY), 50, 255, cv.THRESH_BINARY_INV)[1]
    img[0:desired_height, 0:desired_width] = header
    img = cv.bitwise_and(img, cv.cvtColor(mask, cv.COLOR_GRAY2BGR))
    img = cv.bitwise_or(img, image_canvas)

    cv.imshow("image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
