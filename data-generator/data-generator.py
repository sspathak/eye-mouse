import cv2
import time
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
global frame
if vc.isOpened(): # try to get the first frame
    # global frame
    rval, frame = vc.read()
    vc.read()
    vc.read()
else:
    rval = False

count = 0

#manually edit x and y while collecting data
x=0
y=0
time.sleep(5)
while rval:
    cv2.imshow("preview", frame)
    # input(frame)
    cv2.imwrite("./trash/frame-%d-%d-%d.jpg" % (x, y, count), frame)
    count += 1
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    if count > 300:
        break
cv2.destroyWindow("preview")