from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Bidirectional, LSTM, Dropout

MAX_FRAMES = 10

def prepare_and_create_xception(self):
  # Create a pre-trained Xception model
  base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))

  # Freeze the Xception model layers
  for layer in base_model.layers:
      layer.trainable = False

  return base_model

def create() -> None:
  base_model = prepare_and_create_xception()
  main_model = Sequential()
  adam = Adam(learning_rate=0.001)

  # Add a pre-trained Xception model
  main_model.add(TimeDistributed(base_model, input_shape=(MAX_FRAMES, 299, 299, 3)))
  main_model.add(TimeDistributed(Flatten()))

  # Add a Bi-LSTM layer
  main_model.add(Bidirectional(LSTM(64, return_sequences=True)))
  main_model.add(Bidirectional(LSTM(64, return_sequences=True)))

  # Add a Dropout
  main_model.add(Dropout(0.4))

  # Add a classification layer
  main_model.add(Dense(1024, activation="relu"))
  main_model.add(Dense(1024, activation="hard_sigmoid"))
  main_model.add(Dense(units=1, activation="sigmoid"))

  # Compile the model
  main_model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

  return main_model
