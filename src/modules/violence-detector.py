from datetime import date
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten, Dense, Bidirectional, LSTM, Dropout
from keras.callbacks import ModelCheckpoint
from livelossplot.inputs.keras import PlotLossesCallback
from keras.utils import np_utils
# from keras_attention import Attention
from keras.applications.xception import Xception
from keras.metrics import Recall
from keras.optimizers import adam_v2

from constants import BATCH_SIZE, EPOCHS

class ViolenceDetector:
  def __init__(self) -> None:
    self.model = Sequential()
    self.label_vocabulary = ['fight', 'noFight']

  def prepare_and_create_xception(self):
    # Create a pre-trained Xception model
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))

    # Freeze the Xception model layers
    for layer in base_model.layers:
        layer.trainable = False

    return base_model

  def create(self) -> None:
    base_model = self.prepare_and_create_xception()
    adam = adam_v2.Adam(learning_rate=0.001)

    # Add a pre-trained Xception model
    self.model.add(TimeDistributed(base_model, input_shape=(10, 299, 299, 3)))
    self.model.add(TimeDistributed(Flatten()))

    # Add a Bi-LSTM layer
    self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
    self.model.add(Bidirectional(LSTM(64, return_sequences=True)))

    # Add a Dropout
    self.model.add(Dropout(0.4))

    # Add a Attention layer
    # self.model.add(Attention())

    self.model.add(Dense(1024, activation="relu"))
    self.model.add(Dense(1024, activation="hard_sigmoid"))

    # Add a classification layer
    self.model.add(Dense(2, activation="softmax"))

    # Compile the model
    self.model.compile(loss="mean_squared_error", optimizer=adam, metrics=["accuracy", Recall()])

  def run(self):
    weights_filepath = f"weights/best-weights-{str(date.today())}.h5"

    # Config callback's
    checkpoint = ModelCheckpoint(
        weights_filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    plot_loss = PlotLossesCallback()

    train_labels = test_labels = np_utils.to_categorical(self.label_vocabulary, 2)

    history = self.model.fit(
      # [train_data[0],train_data[1]],
      train_labels,
      validation_split=0.1,
      epochs=EPOCHS,
      callbacks=[checkpoint, plot_loss],
      batch_size=BATCH_SIZE
    )

violence_detector = ViolenceDetector()
violence_detector.create()
