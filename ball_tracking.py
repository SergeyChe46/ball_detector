from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# нижняя и верхняя граница цветов для отслеживаемого объекта
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# ели был предоставлен путь для видеофайла
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# используется камера
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)
while True:
    # считываем текущий кадр
    frame = vs.read()
    # обрабатываем кадр в зависимости от видео/камеры
    frame = frame[1] if args.get("video", False) else frame
    # если кадра нет - выходим
    if frame is None:
        break
    # устанавливаем размер захваченного кадра, блюрим его  и переводим в HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # с помощбю замыкания подготавливаем маску для отслеживаемого объекта
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # ищем контуры созданной маски и её центр
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # ели был найден хотя бы 1 контур
    if len(cnts) > 0:
        # ищем наибольший контур
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # рисуем окружность контура и точку в центре
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # обновляем список точек для "хвоста"
    pts.appendleft(center)

    # рисуем "хвост"
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()
# close all windows
cv2.destroyAllWindows()
