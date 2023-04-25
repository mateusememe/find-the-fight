import sys
import cv2
import numpy as np

sys.path.append("src//modules")

import imageutils

img = cv2.imread('./src/test/images/persons-5.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def person_detection_with_hog():
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur_pm = imageutils.PM_anisotropic_diffusion(img, 20, 0.20, 0.2)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  unsharp = cv2.addWeighted(gray, 2.0, blur, -1.0, 1)

  (regions_normal, _) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.075)
  (regions_gray, _) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.075)
  (regions_blur_pm, _) = hog.detectMultiScale(blur_pm, winStride=(4, 4), padding=(8, 8), scale=1.075)
  (regions_blur, _) = hog.detectMultiScale(blur, winStride=(4, 4), padding=(8, 8), scale=1.075)
  (regions_unsharp, _) = hog.detectMultiScale(unsharp, winStride=(4, 4), padding=(8, 8), scale=1.075)

  count_regions_normal = len(regions_normal)
  count_regions_gray = len(regions_gray)
  count_regions_blur_pm = len(regions_blur_pm)
  count_regions_blur = len(regions_blur)
  count_regions_unsharp = len(regions_unsharp)

  print(f"number of people [regions_normal]: {count_regions_normal}\n")
  print(f"number of people [regions_gray]: {count_regions_gray}\n")
  print(f"number of people [regions_blur_pm]: {count_regions_blur_pm}\n")
  print(f"number of people [regions_blur]: {count_regions_blur}\n")
  print(f"number of people [regions_unsharp]: {count_regions_unsharp}\n")

  for (x, y, w, h) in regions_normal:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


  for (x, y, w, h) in regions_gray:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)


  for (x, y, w, h) in regions_blur_pm:
    cv2.rectangle(blur_pm, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(blur_pm, (x, y), (x + w, y + h), (0, 0, 255), 2)


  for (x, y, w, h) in regions_blur:
    cv2.rectangle(blur, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(blur, (x, y), (x + w, y + h), (0, 0, 255), 2)


  for (x, y, w, h) in regions_unsharp:
    cv2.rectangle(unsharp, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(unsharp, (x, y), (x + w, y + h), (0, 0, 255), 2)

  # cv2.imshow('image_normal', cv2.resize(img, (512,512)))
  # cv2.imshow('image_gray', cv2.resize(gray, (512,512)))
  cv2.imshow('image_blur_pm', cv2.resize(blur_pm, (512,512)))
  # cv2.imshow('image_blur', cv2.resize(blur, (512,512)))
  # cv2.imshow('image_unsharp', cv2.resize(unsharp, (512,512)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Não FUNCIONA AINDA...
classifier = cv2.CascadeClassifier('.//src//test//haarcascade_fullbody.xml')
rects = classifier.detectMultiScale(gray, 1.3, 5)

print(rects)

for (x, y, w, h) in rects:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('image', gray)
cv2.waitKey(0)
