import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src//modules")

import imageutils
from preprocessing import Preprocessing

pp_images = Preprocessing(dataset_path="VIOLENCE_DETECTION_DATASET_PATH")
frames = pp_images.load_video(path="violence-6.mp4", class_subpath="violence", max_frames=6)

gray = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
blur_pm = imageutils.PM_anisotropic_diffusion(frames[5], 20, 0.20, 0.2)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
unsharp = cv2.addWeighted(gray, 2.0, blur, -1.0, 1)

proto_file = 'src\\pretreined\\openpose\\pose_deploy.prototxt'
weights_file = 'src\\pretreined\\openpose\\pose_iter_584000.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

image = frames[5]
frameHeight ,frameWidth, _ = image.shape

# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)

inpBlob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(inWidth, inHeight),
                          mean=(0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()

i = 0
probMap = output[0, i, :, :]
probMap = cv2.resize(probMap, (frameWidth, frameHeight))

mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.imshow("image", probMap)

cv2.waitKey(0)
cv2.destroyAllWindows()
