import cv2
import sys
import numpy as np

sys.path.append("src//modules")

import imageutils
from preprocessing import Preprocessing
# from modules.constants import DATASETS_PATH

pp_images = Preprocessing(dataset_path="VIOLENCE_DETECTION_DATASET_PATH")
frames = pp_images.load_video(path="violence-6.mp4", class_subpath="violence", max_frames=6)

# imageutils.show_image_list(list_images=list(frames),
#                             num_cols=3,
#                             figsize=(9, 9),
#                             grid=False,
#                             title_fontsize=20)
gray = cv2.cvtColor(frames[5], cv2.COLOR_BGR2GRAY)
blur_pm = imageutils.PM_anisotropic_diffusion(frames[5], 20, 0.20, 0.2)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
unsharp = cv2.addWeighted(gray, 2.0, blur, -1.0, 1)

# imageutils.show_image_list(list_images=[ gray, blur_pm, blur, unsharp ], 
#                 list_titles=['Gray', 'Malik', 'Gaussian Blur', 'Sharpened'],
#                 num_cols=4,
#                 figsize=(12, 12),
#                 grid=False,
#                 title_fontsize=10)

proto_file = 'src\\pretreined\\openpose\\pose_deploy.prototxt'
model_file = 'src\\pretreined\\openpose\\pose_iter_584000.caffemodel'

image = frames[5]
input_blob = cv2.dnn.blobFromImage(image, scalefactor=1.0/255, size=(368, 368), mean=(0, 0, 0), swapRB=True, crop=False)

net = cv2.dnn.readNetFromCaffe(proto_file, model_file)

net.setInput(input_blob)

output = net.forward()

print(output.shape)

# Get the number of detected people
num_people = output.shape[0]

H = output.shape[2]
W = output.shape[3]

# Empty list to store the keypoints
points = []

for i in range(25):
  # Get the confidence score for the ith keypoint
  confidence = output[0, i, :, :]

  # Find the (x, y) coordinates of the keypoint with the highest confidence score
  x, y = np.unravel_index(confidence.argmax(), confidence.shape)

  # Scale the (x, y) coordinates to the size of the input image
  x = x / (W - 1) * image.shape[1]
  y = y / (H - 1) * image.shape[0]

  # Add the keypoint coordinates to the list
  points.append((int(x), int(y)))


for i in range(len(points)):
  cv2.circle(image, points[i], 4, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# input_blob = cv2.dnn.blobFromImage(image, 1.0, (368, 368), (0, 0, 0), swapRB=True, crop=True)
# for i in range(output.shape[0]):
#     keypoints = []
#     for j in range(output.shape[1]):
#         confidence = output[i][j][2]
#         if confidence.any() > 0.2:
#             x = int(output[i][j][0][0] * image_width)
#             y = int(output[i][j][1][1] * image_height)
#             keypoints.append((x, y))

#     for x, y in keypoints:
#         cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
# # Define the keypoints of COCO dataset
# keypoints_list = [
#   "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", 
#   "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", 
#   "LEye", "REar", "LEar"
# ]
# image = cv2.resize(image, (368, 368))

# # Define the connections between keypoints
# connections = [
#   [0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [9,10],
#   [10,11], [8,12], [12,13], [13,14], [0,15], [15,17], [0,16], [16,18]
# ]

# for i in range(num_people):
#     keypoints = output[i]
#     print(keypoints.shape)
#     for j in range(keypoints.shape[0]):
#         keypoint = keypoints[j]
#         # print(keypoint[0].shape)
#         # print(keypoint[1].shape)
#         # print(keypoint[2].shape)

#         x = int(keypoint[0][0])
#         y = int(keypoint[1][1])
#         print(x,y)
#         score = (keypoint[2])

#         if score > 0.5:
#             cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

# # Loop over the detected people and plot the keypoints on the image
# for i in range(num_people):
#     # Create an empty numpy array to store the keypoints
#     keypoints = np.zeros((18, 3))
#     # Fill the numpy array with the detected keypoints
#     for j in range(len(keypoints_list)):
#       print(output[i][j][0][0], image.shape[1])
#       keypoint_confidence = output[i][j][2][0]
#       keypoint_x = int(output[i][j][0][0] * image.shape[1])
#       keypoint_y = int(output[i][j][1][1] * image.shape[0])
#       print([keypoint_x, keypoint_y, keypoint_confidence])
#       keypoints[j] = [keypoint_x, keypoint_y, keypoint_confidence]

#     # Loop over the connections and draw them on the image
#     for connection in connections:
#         start_idx, end_idx = connection
#         start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
#         end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
#         cv2.line(image, start_point, end_point, (0, 255, 0), thickness=2)

#     # # Loop over the keypoints and draw them on the image
#     # for k in range(len(keypoints)):
#     #   keypoint_x = int(keypoints[k][0])
#     #   keypoint_y = int(keypoints[k][1])
#     #   keypoint_confidence = keypoints[k][2]

# cv2.imshow("openpose", image)

