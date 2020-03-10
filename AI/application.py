from keras.models import load_model
model = load_model("models/simple_CNN_100_acc.h5")
import cv2
import numpy as np
import time
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_face(filename, required_size=(224, 224), image_array=None):
    global detector
    if image_array is None:
        pixels = pyplot.imread(filename)
    else:
        pixels = np.array(image_array)
    results = detector.detect_faces(pixels)
    try:
        left_eye_x, left_eye_y = results[0]['keypoints']['left_eye']
        right_eye_x, right_eye_y = results[0]['keypoints']['right_eye']
    except IndexError:
        return None

    left_eye_image = pixels[left_eye_y - 25:left_eye_y + 25, left_eye_x - 50:left_eye_x + 50]
    right_eye_image = pixels[right_eye_y - 25:right_eye_y + 25, right_eye_x - 50:right_eye_x + 50]
    eyes_img = cv2.hconcat([left_eye_image, right_eye_image])
    image = Image.fromarray(eyes_img)
    face_array = asarray(image)
    return face_array


def display_coords(target_coords):
    global blank
    y = 0 if target_coords < 3 else 1
    x = target_coords if y == 0 else target_coords - 3
    img = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])

    img[y][x] = 1 if target_coords != -1 else 0
    return img



pyplot.gca().set_axis_off()
pyplot.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
pyplot.margins(0,0)
pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())


img_obj = pyplot.imshow(display_coords(0))
pyplot.draw()


detector = MTCNN()
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

x = 475
y = 250
h = 50
w = 200
time.sleep(5)

while rval:

    count += 1
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    img = extract_face("", image_array=frame)
    if img is None:
        img_obj.set_data(display_coords(-1))
        pyplot.draw()
        pyplot.pause(0.0001)
        continue

    cv2.imshow("preview", img)
    im = img.reshape(1, h, w, 3)
    prediction = model.predict(im)
    print([int(i*100) for i in prediction[0]], end=" ")
    target_coords = np.argmax(prediction)
    print(target_coords)
    img_obj.set_data(display_coords(target_coords))
    pyplot.draw()
    pyplot.pause(0.01)
    time.sleep(0.01)

cv2.destroyWindow("preview")