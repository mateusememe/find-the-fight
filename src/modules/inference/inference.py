from modules.TrainModel.train_model import TrainModel
from keras.layers import StringLookup, Dropout, Dense, Bidirectional, LSTM, Layer, GlobalAveragePooling2D, Attention
from livelossplot.inputs.keras import PlotLossesCallback
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from google.colab.patches import cv2_imshow
from keras.applications import xception
from wandb.keras import WandbCallback
from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from keras import Input, Model
from imutils import paths
from copy import deepcopy
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np

class Inference:
  def __init__(self) -> None:
    self.IMG_SIZE = 299
    self.BATCH_SIZE = 10
    self.EPOCHS = 25
    self.MAX_SEQ_LENGTH = 10
    self.NUM_FEATURES = 2048
    self.fight_model = TrainModel()

  def prepare_single_video(self, frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, self.MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(self.MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = self.fight_model.feature_extractor.predict(batch[None, j, :], verbose=0)
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask
