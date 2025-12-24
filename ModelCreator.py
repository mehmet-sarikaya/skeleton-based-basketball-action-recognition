import tensorflow as tf
from tensorflow.keras import models,layers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM


class ModelCreator:
    def __init__(self, win_len_sec=3, fps=10, num_classes=3):
        self.model = None
        self.duration_sec = win_len_sec
        self.fps = fps
        # 1.number of frames per window, 2. Keypoints per frame, 3.X/Y Coordinates, 4.Only 1 Channel (NO RGB)
        self.input_shape = (win_len_sec * fps, 17, 2, 1)
        self.num_classes = num_classes

    def create_conv2d_lstm_model(self):
        self.model = models.Sequential([
            # Input Shape: (Timesteps, Height/Keypoints, Width/Coords, Channels)
            # (30 frames, 17 keypoints, 2 coords, 1 channel)
            layers.Input(shape=self.input_shape),

            # 1. Feature Extraction (Spatial)
            # Apply Conv2D to each frame individually
            layers.TimeDistributed(layers.Conv2D(32, (3, 2), padding='same', activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 1))),
            layers.TimeDistributed(layers.Conv2D(64, (3, 1), padding='same', activation='relu')),
            layers.TimeDistributed(layers.Flatten()),

            # 2. Sequence Learning (Temporal)
            # The Flattened output of the CNNs goes into the LSTM
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.1),
            layers.LSTM(32),

            # 3. Classification
            layers.Dense(32, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return self.model

    def create_small_conv2d_lstm_model(self):
        self.model = models.Sequential([
            layers.Input(shape=self.input_shape),

            layers.TimeDistributed(layers.Conv2D(16, (3, 2), padding='same', activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 1))),
            layers.TimeDistributed(layers.Flatten()),

            layers.LSTM(32),
            layers.Dropout(0.3),

            # 3. Classification
            layers.Dense(16, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model

    def create_cnn_lstm_model(self):
        self.model = models.Sequential(
            name="CNN_LSTM_Skeleton_Model",
            layers=[
                # Input layer:
                # Using hp_units=128, hp_l2=0.001, hp_dropout=0.1, hp_activation='relu'
                Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.4),

                Conv1D(128, kernel_size=3, padding='same', activation='relu'),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.4),

                LSTM(128, return_sequences=True),
                Dropout(0.4),
                LSTM(256),
                Dropout(0.4),

                # Output layer: softmax for multi-class classification
                Dense(self.num_classes, activation='softmax')
            ]
        )

    # https://doi.org/10.1109/ACCESS.2021.3125733
    def create_industry_4_model(self):
        self.model = models.Sequential(
            name="industry 4.0 model",
            layers=[
                # 1. Input Layer & First Convolutional Layer
                # Input shape: (90 time steps, 3 features)
                layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(90, 3)),

                # 2. Max-Pooling Layer
                # Reduces spatial dimensions (90 -> 45)
                layers.MaxPooling1D(pool_size=2),

                # 3. Second Convolutional Layer
                # Output shape shows a depth of 32
                layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),

                # 4. Flatten Layer
                # Converts 45 x 32 into a 1D vector of 1440
                layers.Flatten(),

                # 5. Dense (Fully Connected) Layer
                # Output shape: 6
                layers.Dense(6, activation='relu'),

                # 6. Output Layer
                # Typically uses Softmax for multi-class classification
                layers.Dense(6, activation='softmax')
            ]
        )
