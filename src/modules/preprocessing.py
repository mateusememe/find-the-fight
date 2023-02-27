import os
from constants import MAX_FRAMES, IMG_SHAPE, DATASETS_PATH
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
    self.dataset_path = DATASETS_PATH[dataset_path]
    self.output_path = f"{self.dataset_path}\\data"

  def crop_center_square(self, frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

  def load_video(self, path: str, class_subpath: str):
    temp_path = f"{self.dataset_path}\\{path}"
    cap = cv2.VideoCapture(f"{self.dataset_path}\\{class_subpath}\\{path}")
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

  def rename_files(self):
    violence_videos_paths = list(paths.list_files(f"{self.dataset_path}\\violence"))
    non_violence_videos_paths = list(paths.list_files(f"{self.dataset_path}\\nonViolence"))

    for index, violence_path in enumerate(violence_videos_paths):
      video_name = violence_path.split("\\")[-1]
      video_extension = video_name.split(".")[-1]
      new_violence_path = violence_path.replace(video_name, f"violence-{index+1}.{video_extension}")

      print(new_violence_path)
      os.rename(violence_path, new_violence_path)

    for index, violence_path in enumerate(non_violence_videos_paths):
      video_name = violence_path.split("\\")[-1]
      video_extension = video_name.split(".")[-1]
      new_non_violence_path = violence_path.replace(video_name, f"non-violence-{index+1}.{video_extension}")

      print(new_non_violence_path)
      os.rename(violence_path, new_non_violence_path)

  def prepare_dataframe(self):
    violence_videos_paths = list(paths.list_files(f"{self.dataset_path}\\violence"))
    non_violence_videos_paths = list(paths.list_files(f"{self.dataset_path}\\non-violence"))

    extract_class = lambda video_name: video_name.split(self.dataset_path)[1].split("\\")[1]
    extract_video_name = lambda video_name: video_name.split(self.dataset_path)[1].split("\\")[2]

    all_videos_path = violence_videos_paths + non_violence_videos_paths

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


    print(f"\nTotal Videos: {len(os.listdir(self.output_path))}")

  def load_data(self):
    dataset_name = self.dataset_path.split("\\")[-1]
    print(f"\n\n### Selected Dataset: {dataset_name}")

    self.prepare_dataframe()
    print("\n\t\t     Dataset HEAD")
    print(self.dataset.head())

    self.move_videos()

# preprocessing = Preprocessing(dataset_path="VIOLENCE_DETECTION_DATASET_PATH")

# preprocessing.load_data()
# preprocessing.rename_files() for VIOLENCE_DETECTION_DATASET_PATH
