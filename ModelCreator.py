import tensorflow as tf
from tensorflow.keras import models,layers
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, LSTM, TimeDistributed, Reshape, Flatten


class ModelCreator:
    def __init__(self, model_name, win_len_sec, fps, num_classes):
        self.model = None
        self.duration_sec = win_len_sec
        self.fps = fps
        # 1.number of frames per window, 2. Keypoints per frame, 3.X/Y Coordinates, 4.Only 1 Channel (NO RGB)
        self.input_shape = (win_len_sec * fps, 12, 2, 1)
        self.num_classes = num_classes

        self.model_name = model_name

        self.model_func_dic = {
            "conv2d": self.create_conv2d_lstm_model,
            "conv2d_s": self.create_small_conv2d_lstm_model,
            "cnn_lstm": self.create_cnn_lstm_model,
            "ind_4": self.create_industry_4_model,
            "small_cnn": self.create_small_robust_cnn,
            "small_cnn_lstm": self.create_small_robust_cnn_lstm
        }

    def create_model(self):
        if self.model_name not in self.model_func_dic:
            raise ValueError(f"Model Name: {self.model_name} unbekannt")
        func_to_create = self.model_func_dic[self.model_name]
        return func_to_create()

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
                # Step 1: Flatten (12, 2, 1) -> 24 but keep 120 frames
                # Input: (None, 120, 12, 2, 1) -> Output: (None, 120, 24)
                layers.Reshape((-1, 24), input_shape=self.input_shape),

                # Step 2: Conv1D slides over the 120 frames (Time)
                layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Dropout(0.4),

                layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),

                # Step 3: LSTM processes the reduced sequence
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.4),
                layers.LSTM(256),
                layers.Dropout(0.4),

                layers.Dense(self.num_classes, activation='softmax')
            ]
        )
        return self.model

    # https://doi.org/10.1109/ACCESS.2021.3125733
    def create_industry_4_model(self):
        self.model = models.Sequential(
            name="industry_4.0_model",
            layers=[
                layers.Reshape((-1, 24), input_shape=self.input_shape),

                # 1. Input Layer & First Convolutional Layer
                # Input shape: (90 time steps, 3 features)
                layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=self.input_shape),

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
                layers.Dense(self.num_classes, activation='relu'),

                # 6. Output Layer
                # Typically uses Softmax for multi-class classification
                layers.Dense(self.num_classes, activation='softmax')
            ]
        )

        return self.model

    def create_small_robust_cnn(self):
        """
        Optimiert für kleine Datensätze:
        - Wenig Filter (reduziert Komplexität)
        - GlobalAveragePooling statt Flatten (verhindert Parameter-Explosion)
        - Hoher Dropout & BatchNormalization (stabile Generalisierung)
        """
        self.model = models.Sequential(
            name="Small_Robust_CNN",
            layers=[
                # (Frames, 12, 2, 1) -> (Frames, 24)
                layers.Reshape((-1, 24), input_shape=self.input_shape),

                layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),  # Reduziert Zeitachse
                layers.Dropout(0.4),

                layers.GlobalAveragePooling1D(),

                layers.Dense(32, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ]
        )
        return self.model

    def create_small_robust_cnn_lstm(self):
        self.model = models.Sequential(
            name="Small_Robust_LSTM",
            layers=[
                layers.Input(shape=self.input_shape),
                layers.Reshape((-1, 24)),

                layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.3),

                layers.LSTM(32, return_sequences=True),
                layers.Dropout(0.3),

                layers.GlobalAveragePooling1D(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ]
        )
        return self.model
