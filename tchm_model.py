import tensorflow as tf
from tensorflow.keras import layers, models


def temporal_block(x, n_filters, kernel_size, dilation_rate):
    skip = x
    # Causal sorgt dafür, dass das Modell nicht in die Zukunft schaut
    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='causal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters=n_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding='causal')(x)
    x = layers.BatchNormalization()(x)

    if skip.shape[-1] != x.shape[-1]:
        skip = layers.Conv1D(n_filters, kernel_size=1, padding='same')(skip)

    x = layers.add([skip, x])
    return layers.Activation('relu')(x)


def build_tcmh_skeleton_model(input_shape, num_classes):
    # Input: (16, 12, 2, 1)
    main_input = layers.Input(shape=input_shape)

    # 1. Daten vorbereiten: (16, 12, 2)
    x = layers.Reshape((input_shape[0], input_shape[1] * input_shape[2]))(main_input)

    # 2. TCN Teil: Extrahiere Merkmale über die Zeit
    # Wir reduzieren die Filter, damit das Modell nicht auswendig lernt (Overfitting)
    x = temporal_block(x, 64, kernel_size=3, dilation_rate=1)
    x = temporal_block(x, 64, kernel_size=3, dilation_rate=2)

    # 3. Multi-Head Attention
    # Wir nutzen nur 2 Köpfe, damit die Aufmerksamkeit fokussierter ist
    # key_dim wird kleiner gewählt
    attn_out = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)

    # 4. Globales Verständnis
    x = layers.GlobalAveragePooling1D()(x)

    # 5. Klassifikation mit stärkerem Dropout
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=main_input, outputs=outputs, name="TCMH_Optimized")