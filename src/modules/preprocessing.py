
import os
from constants import MAX_FRAMES, IMG_SHAPE, VIOLENCE_DETECTION_DATASET_PATH
from imutils import paths
import shutil
import pandas as pd
import numpy as np
import cv2

class Preprocessing:
  def __init__(self, dataset_path: str) -> None:
    self.dataset = None
    self.test_data = None
    self.train_data = None
    self.dataset_path = dataset_path
    self.output_path = f"{dataset_path}\\data"

  def crop_center_square(self, frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

  def load_video(self, path: str):
    cap = cv2.VideoCapture(path)
    video_frames = []

    try:
      while True:
        ret, frame = cap.read()

        if not ret:
          break

        # pick the center square from frame
        frame = self.crop_center_square(frame)

        # resize the frame for model input shape
        frame = cv2.resize(frame, IMG_SHAPE)
        cv2.imshow("", frame)
        cv2.waitKey(0)
        frame = frame[:, :, [2, 1, 0]]

        video_frames.append(frame)

        if len(video_frames) == MAX_FRAMES:
          break
    finally:
        cap.release()

    return np.array(video_frames), video_frames

  def prepare_dataframe(self):
    violent_videos_paths = list(paths.list_files(f"{self.dataset_path}\\violent"))
    non_violent_videos_paths = list(paths.list_files(f"{self.dataset_path}\\non-violent"))

    extract_class = lambda video_name: video_name.split(self.dataset_path)[1].split("\\")[1]
    extract_video_name = lambda video_name: video_name.split(self.dataset_path)[1]

    all_videos_path = violent_videos_paths + non_violent_videos_paths

    self.dataset = pd.DataFrame()

    self.dataset['video_name'] = all_videos_path
    self.dataset['class'] = self.dataset['video_name'].apply(extract_class)
    self.dataset['video_name'] = self.dataset['video_name'].apply(extract_video_name)

  def move_videos(self):
    if not os.path.exists(self.output_path):
      os.mkdir(self.output_path)

    for index, item in self.dataset.iterrows():
      videoFile = item['video_name']
      videoClass = item['class']
      join_path = f"{self.dataset_path}\\{videoClass}\\"
      videoPath = os.path.join(join_path, videoFile)
      shutil.copy2(videoPath, self.output_path)

    print(f"Total videos: {len(os.listdir(self.output_path))}")

  def load_data(self):
    self.prepare_dataframe()

    print(self.dataset.head())

preprocessing = Preprocessing(VIOLENCE_DETECTION_DATASET_PATH, )

preprocessing.load_data()
