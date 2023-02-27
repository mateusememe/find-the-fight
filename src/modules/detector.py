import copy
import numpy as np
import pandas as pd
from datetime import date
from tensorflow.keras import utils
from preprocessing import Preprocessing
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from constants import BATCH_SIZE, EPOCHS, MAX_FRAMES
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Attention, TimeDistributed, Flatten, Dense, Bidirectional, LSTM, Dropout
# from livelossplot.inputs.keras import PlotLossesCallback
# from keras_attention import Attention

class HumanViolenceDetector:
  def __init__(self) -> None:
    self.model = Sequential()
    self.preprocessing = Preprocessing(dataset_path="VIOLENCE_DETECTION_DATASET_PATH")
    self.preprocessing.load_data()
    self.label_vocabulary = None

  def prepare_and_create_xception(self):
    # Create a pre-trained Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))

    # Freeze the Xception model layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

  def create(self) -> None:
    base_model = self.prepare_and_create_xception()
    adam = Adam(learning_rate=0.001)

    # Add a pre-trained Xception model
    self.model.add(TimeDistributed(base_model, input_shape=(MAX_FRAMES, 299, 299, 3)))
    self.model.add(TimeDistributed(Flatten()))

    # Add a Bi-LSTM layer
    self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
    self.model.add(Bidirectional(LSTM(64, return_sequences=True)))

    # Add a Dropout
    self.model.add(Dropout(0.4))

    # Add a Attention layer
    self.model.add(Attention())

    self.model.add(Dense(1024, activation="relu"))
    self.model.add(Dense(1024, activation="hard_sigmoid"))

    # Add a classification layer
    self.model.add(Dense(2, activation="softmax"))

    # Compile the model
    self.model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy", Recall()])

  # TODO: Make a repeat to persist the shape (10, 299, 299, 3) in a list to train the model
  def load_frames_from_videos(self, dataframe: pd.DataFrame):
    tokenizer = Tokenizer()
    video_paths = list(dataframe["video_name"].values)
    tokenizer.fit_on_texts(dataframe["class"].values)
    labels = np.array([[token[0]-1] for token in tokenizer.texts_to_sequences(dataframe["class"].values)])

    out_data = []
    out_labels = []

    for index_video_path in range(len(video_paths)):
      frames = self.preprocessing.load_video(path=video_paths[index_video_path], class_subpath=dataframe["class"].values[index_video_path])
      out_data.append(frames)
      out_labels.append(labels[index_video_path])

    return out_data, out_labels

  def prepare_data(self):
    train_df, test_df = train_test_split(self.preprocessing.dataset, test_size = 0.30, stratify=self.preprocessing.dataset['class'])

    topNActs = train_df["class"].value_counts().nlargest(10).reset_index()["index"].tolist()
    train_new = train_df[train_df["class"].isin(topNActs)]
    test_new = test_df[test_df["class"].isin(topNActs)]

    train_new = train_new.reset_index(drop=True)
    test_new = test_new.reset_index(drop=True)

    self.preprocessing.train_data = copy.deepcopy(train_new)
    self.preprocessing.test_data = copy.deepcopy(test_new)

    self.label_vocabulary = np.unique(self.preprocessing.train_data["class"])

  def run(self, train_data):
    weights_filepath = f"weights/best-weights-{str(date.today())}.h5"

    # Config callback's
    checkpoint = ModelCheckpoint(
        weights_filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    # plot_loss = PlotLossesCallback()

    train_labels = test_labels = utils.to_categorical(self.label_vocabulary, 2)

    history = self.model.fit(
      # [train_data[0],train_data[1]],
      train_labels,
      validation_split=0.1,
      epochs=EPOCHS,
      callbacks=[checkpoint],
      batch_size=BATCH_SIZE
    )

violence_detector = HumanViolenceDetector()
violence_detector.prepare_data()
out_data, out_labels = violence_detector.load_frames_from_videos(dataframe=violence_detector.preprocessing.train_data)

print(f"data length: {len(out_data)}\nlabels length: {len(out_labels)}\n")

