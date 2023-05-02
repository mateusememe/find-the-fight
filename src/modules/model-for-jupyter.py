# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.layers import TimeDistributed, Flatten, Dense, Bidirectional, LSTM, Dropout

# MAX_FRAMES = 10

# def prepare_and_create_xception(self):
#   # Create a pre-trained Xception model
#   base_model = Xception(weights='imagenet', include_top=False, input_shape=(299,299,3))

#   # Freeze the Xception model layers
#   for layer in base_model.layers:
#       layer.trainable = False

#   return base_model

# def create() -> None:
#   base_model = prepare_and_create_xception()
#   main_model = Sequential()
#   adam = Adam(learning_rate=0.001)

#   # Add a pre-trained Xception model
#   main_model.add(TimeDistributed(base_model, input_shape=(MAX_FRAMES, 299, 299, 3)))
#   main_model.add(TimeDistributed(Flatten()))

#   # Add a Bi-LSTM layer
#   main_model.add(Bidirectional(LSTM(64, return_sequences=True)))
#   main_model.add(Bidirectional(LSTM(64, return_sequences=True)))

#   # Add a Dropout
#   main_model.add(Dropout(0.4))

#   # Add a classification layer
#   main_model.add(Dense(1024, activation="relu"))
#   main_model.add(Dense(1024, activation="hard_sigmoid"))
#   main_model.add(Dense(units=1, activation="sigmoid"))

#   # Compile the model
#   main_model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])

#   return main_model
"""
  2023-05-01 16:06:07.412991: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
  To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
  2023-05-01 16:06:09.902021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /device:GPU:0 with 3976 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1660, pci bus id: 0000:01:00.0, compute capability: 7.5
  [
    name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 5850415605982010966
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 4169138176
    locality {
      bus_id: 1
      links {
      }
    }
  incarnation: 14711510699743823726
  physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1660, pci bus id: 0000:01:00.0, compute capability: 7.5"
]


"""

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
