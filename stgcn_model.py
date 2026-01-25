import tensorflow as tf
from stgcn_graph import Graph

REGULARIZER = tf.keras.regularizers.l2(l2=1e-4)
INITIALIZER = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_out", distribution="truncated_normal"
)

class SGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            filters * kernel_size,
            kernel_size=1,
            padding='same',
            kernel_initializer=INITIALIZER,
            data_format='channels_first',
            kernel_regularizer=REGULARIZER
        )

    def call(self, x, A, training=False):
        # x: (N, C, T, V)
        x = self.conv(x)  # (N, filters*K, T, V)

        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]

        x = tf.reshape(x, [N, self.kernel_size, C // self.kernel_size, T, V])
        # (N,K,C,T,V) x (K,V,V) -> (N,C,T,V)
        x = tf.einsum('nkctv,kvw->nctw', x, A)
        return x, A

class STGCN(tf.keras.Model):
    def __init__(self, filters, kernel_size=(9, 3), stride=1, activation='relu',
                 residual=True, downsample=False):
        super().__init__()
        self.sgcn = SGCN(filters, kernel_size=kernel_size[1])

        self.tgcn = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(
                filters,
                kernel_size=(kernel_size[0], 1),
                strides=(stride, 1),
                padding='same',
                kernel_initializer=INITIALIZER,
                data_format='channels_first',
                kernel_regularizer=REGULARIZER
            ),
            tf.keras.layers.BatchNormalization(axis=1),
        ])

        self.act = tf.keras.layers.Activation(activation)

        if not residual:
            self.residual = lambda x, training=False: 0
        elif stride == 1 and not downsample:
            self.residual = lambda x, training=False: x
        else:
            self.residual = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=(1, 1),
                    strides=(stride, 1),
                    padding='same',
                    kernel_initializer=INITIALIZER,
                    data_format='channels_first',
                    kernel_regularizer=REGULARIZER
                ),
                tf.keras.layers.BatchNormalization(axis=1),
            ])

    def call(self, x, A, training=False):
        res = self.residual(x, training=training)
        x, A = self.sgcn(x, A, training=training)
        x = self.tgcn(x, training=training)
        x = x + res
        x = self.act(x)
        return x, A

class STGCNClassifier(tf.keras.Model):
    """
    Input: (N, C, T, V, M)
    Output: (N, num_classes)
    """
    def __init__(self, num_classes, num_nodes, edges, in_channels=3):
        super().__init__()

        graph = Graph(num_node=num_nodes, edges=edges, labeling_mode='spatial')
        self.A = tf.Variable(graph.A, dtype=tf.float32, trainable=False, name='A')

        self.in_channels = in_channels
        self.data_bn = tf.keras.layers.BatchNormalization(axis=1)

        self.layers_stgcn = [
            STGCN(64, residual=False),
            STGCN(64),
            STGCN(64),
            STGCN(64),
            STGCN(128, stride=2, downsample=True),
            STGCN(128),
            STGCN(128),
            STGCN(256, stride=2, downsample=True),
            STGCN(256),
            STGCN(256),
        ]

        self.pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        self.logits = tf.keras.layers.Conv2D(
            num_classes,
            kernel_size=1,
            padding='same',
            kernel_initializer=INITIALIZER,
            data_format='channels_first',
            kernel_regularizer=REGULARIZER
        )

    def call(self, x, training=False):
        # x: (N, C, T, V, M)
        N = tf.shape(x)[0]
        C = tf.shape(x)[1]
        T = tf.shape(x)[2]
        V = tf.shape(x)[3]
        M = tf.shape(x)[4]

        # exakt wie im GitHub-Code
        x = tf.transpose(x, perm=[0, 4, 3, 1, 2])      # (N, M, V, C, T)
        x = tf.reshape(x, [N * M, V * C, T])           # (N*M, V*C, T)
        x = self.data_bn(x, training=training)
        x = tf.reshape(x, [N, M, V, C, T])             # (N, M, V, C, T)
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])      # (N, M, C, T, V)
        x = tf.reshape(x, [N * M, C, T, V])            # (N*M, C, T, V)

        A = self.A
        for layer in self.layers_stgcn:
            x, A = layer(x, A, training=training)

        # (N*M, C', T', V) -> pool -> (N*M, C')
        x = self.pool(x)

        # back to (N, M, C', 1, 1) and average over persons
        x = tf.reshape(x, [N, M, -1, 1, 1])
        x = tf.reduce_mean(x, axis=1)

        x = self.logits(x)          # (N, num_classes, 1, 1)
        x = tf.reshape(x, [N, -1])  # (N, num_classes)
        return x
