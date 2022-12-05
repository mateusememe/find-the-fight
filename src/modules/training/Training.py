from keras.layers import StringLookup, Dropout, Dense, Bidirectional, LSTM, Layer, GlobalAveragePooling2D, Attention
from livelossplot.inputs.keras import PlotLossesCallback
from keras.callbacks import ModelCheckpoint
from keras.applications import xception
from keras.optimizers import adam_v2
from keras.metrics import Recall
from keras.utils import np_utils
from keras import backend as K
from keras import Input, Model
import numpy as np
import cv2
from hyperparams import *
class attention(Layer):
  def __init__(self,**kwargs):
    super(attention,self).__init__(**kwargs)

  def build(self,input_shape):
    self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1),
                            initializer='random_normal', trainable=True)
    self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1),
                            initializer='zeros', trainable=True)
    super(attention, self).build(input_shape)

  def call(self,x):
    # Alignment scores. Pass them through tanh function
    e = K.tanh(K.dot(x,self.W)+self.b)
    # Remove dimension of size 1
    e = K.squeeze(e, axis=-1)
    # Compute the weights
    alpha = K.softmax(e)
    # Reshape to tensorFlow format
    alpha = K.expand_dims(alpha, axis=-1)
    # Compute the context vector
    context = x * alpha
    context = K.sum(context, axis=1)
    return context

class Training:
  def __init__(self) -> None:
    self.fight_model = None
    self.train_labels = None
    self.train_data = None
    self.IMG_SIZE = 299
    self.BATCH_SIZE = 10
    self.EPOCHS = 25
    self.MAX_SEQ_LENGTH = 10
    self.NUM_FEATURES = 2048
    self.adam = adam_v2.Adam(learning_rate=0.001)
    self.feature_extractor = self.build_feature_extractor()
    self.label_processor = StringLookup(
      num_oov_indices=0, vocabulary=['fight', 'noFight']
    )

  def build_feature_extractor(self):
    feature_extractor = xception.Xception(
        pooling="max",
        include_top=False,
        weights='imagenet',
        classifier_activation='softmax',
        input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
    )

    preprocess_input  = xception.preprocess_input

    inputs = Input((self.IMG_SIZE, self.IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    return Model(inputs, outputs, name="feature_extractor")

  # Funções básicas para o pré-processamento
  def crop_center_square(self, frame):
      y, x = frame.shape[0:2]
      min_dim = min(y, x)
      start_x = (x // 2) - (min_dim // 2)
      start_y = (y // 2) - (min_dim // 2)
      return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


  def load_video(self, path, max_frames=0, resize=(299, 299)):
      cap = cv2.VideoCapture(path)
      frames = []
      try:
          while True:
              ret, frame = cap.read()
              if not ret:
                  break
              frame = self.crop_center_square(frame)
              frame = cv2.resize(frame, resize)
              frame = frame[:, :, [2, 1, 0]]
              frames.append(frame)
              if len(frames) == max_frames:
                  break
      finally:
          cap.release()
      return np.array(frames)

  # Utility for our sequence model.
  def get_fight_detection_model(self):
    class_vocab = self.label_processor.get_vocabulary()

    frame_features_input = Input((self.MAX_SEQ_LENGTH, self.NUM_FEATURES))
    mask_input = Input((self.MAX_SEQ_LENGTH,), dtype="bool")

    # Bi-LSTM
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(units=64, return_sequences=True))(
        frame_features_input, mask=mask_input
    )
    x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
    x = attention()(x)
    x = Dropout(0.4)(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dense(units=1024, activation="hard_sigmoid")(x)
    output = Dense(len(class_vocab), activation="softmax")(x)

    figh_model = Model([frame_features_input, mask_input], output)

    figh_model.compile(loss="mean_squared_error", optimizer=self.adam, metrics=["accuracy", Recall()])

    return figh_model

  def run_experiment(self, weights_filepath):
    # Config callback's
    checkpoint = ModelCheckpoint(
      weights_filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    plot_loss_1 = PlotLossesCallback()

    self.train_labels_ohe = np_utils.to_categorical(self.train_labels, 2)
    # test_labels_ohe = np_utils.to_categorical(self.test_labels, 2)

    self.fight_model = self.get_self.fight_model()

    history = self.fight_model.fit(
      [self.train_data[0], self.train_data[1]],
      self.train_labels_ohe,
      validation_split=0.1,
      epochs = self.EPOCHS,
      callbacks = [checkpoint, plot_loss_1],
      batch_size = self.BATCH_SIZE
    )

    self.fight_model.load_weights(weights_filepath)
    # _, accuracy = fight_detection_model.evaluate([test_data[0], test_data[1]], test_labels_ohe)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history
