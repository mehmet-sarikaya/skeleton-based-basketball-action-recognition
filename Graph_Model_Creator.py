import tensorflow as tf
from tensorflow.keras import layers, models


def normalize_adjacency(A):
    """A_hat = D^{-1/2} (A + I) D^{-1/2}"""
    A = tf.cast(A, tf.float32)
    V = tf.shape(A)[0]
    I = tf.eye(V, dtype=tf.float32)
    A_tilde = A + I
    d = tf.reduce_sum(A_tilde, axis=1)
    d_inv_sqrt = tf.linalg.diag(tf.pow(d, -0.5))
    return d_inv_sqrt @ A_tilde @ d_inv_sqrt


class GraphConv(layers.Layer):
    """
    Spatial graph convolution per timestep:
      Y[b,t,v,f] = sum_u A_hat[v,u] * (X[b,t,u,c] @ W[c,f])
    Input:  X [B,T,V,C]
    Output: Y [B,T,V,F]
    """
    def __init__(self, out_channels, A_hat, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.A_hat = tf.constant(A_hat, dtype=tf.float32)  # store internally

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.W = self.add_weight(
            shape=(in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )
        self.b = self.add_weight(
            shape=(self.out_channels,),
            initializer="zeros",
            trainable=True,
            name="b",
        )

    def call(self, x, training=False):
        # x: [B,T,V,C]
        y = tf.tensordot(x, self.W, axes=1) + self.b         # [B,T,V,F]
        y = tf.einsum("vu,btuf->btvf", self.A_hat, y)        # aggregate over joints
        return y


class STGCNBlock(layers.Layer):
    """
    ST-GCN block: Spatial GCN -> BN/ReLU -> Temporal Conv -> BN -> Residual -> ReLU -> Dropout
    """
    def __init__(self, out_channels, A_hat, temporal_kernel=9, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.temporal_kernel = temporal_kernel
        self.dropout = dropout

        self.gcn = GraphConv(out_channels, A_hat)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.tcn = layers.Conv2D(
            filters=out_channels,
            kernel_size=(temporal_kernel, 1),
            padding="same",
            use_bias=False,
        )
        self.bn2 = layers.BatchNormalization()
        self.drop = layers.Dropout(dropout)
        self.relu2 = layers.ReLU()

        self.res_conv = None
        self.res_bn = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        if in_channels != self.out_channels:
            self.res_conv = layers.Conv2D(
                filters=self.out_channels,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
            )
            self.res_bn = layers.BatchNormalization()

    def call(self, x, training=False):
        y = self.gcn(x, training=training)
        y = self.bn1(y, training=training)
        y = self.relu1(y)

        y = self.tcn(y)
        y = self.bn2(y, training=training)

        r = x
        if self.res_conv is not None:
            r = self.res_conv(r)
            r = self.res_bn(r, training=training)

        y = y + r
        y = self.relu2(y)
        y = self.drop(y, training=training)
        return y


def build_stgcn_12kp(input_shape, num_classes, edges, temporal_kernel=9):
    """
    input_shape: (T, 12, 2, 1) exactly like your models
    edges: list of undirected edges over your 12 keypoints (index pairs)
    """
    T, V, XYZ, CH = input_shape
    assert V == 12, "Expected 12 keypoints"
    assert XYZ == 3, "Expected (x,y,conf)"
    assert CH == 1, "Expected channel=1"

    # Build adjacency A [V,V] from edges
    A = tf.zeros((V, V), dtype=tf.float32)
    undirected = edges + [(j, i) for (i, j) in edges]
    idx = tf.constant(undirected, dtype=tf.int32)
    A = tf.tensor_scatter_nd_update(A, idx, tf.ones((len(undirected),), tf.float32))
    A_hat = normalize_adjacency(A)

    x_in = layers.Input(shape=input_shape, name="pose_input")      # [B,T,12,2,1]

    # Match your pipeline: squeeze channel -> [B,T,12,2]
    x = layers.Lambda(lambda z: tf.squeeze(z, axis=-1), name="squeeze_channel")(x_in)

    # Optional: normalize per-sample scale (simple BN)
    x = layers.BatchNormalization(name="input_bn")(x)

    # Backbone
    x = STGCNBlock(64, A_hat, temporal_kernel=temporal_kernel, dropout=0.2, name="stgcn1")(x)
    x = STGCNBlock(64, A_hat, temporal_kernel=temporal_kernel, dropout=0.2, name="stgcn2")(x)
    x = STGCNBlock(128, A_hat, temporal_kernel=temporal_kernel, dropout=0.3, name="stgcn3")(x)
    x = STGCNBlock(256, A_hat, temporal_kernel=temporal_kernel, dropout=0.4, name="stgcn4")(x)

    # Global pooling over time + joints -> [B,256]
    x = layers.Lambda(lambda z: tf.reduce_mean(z, axis=[1, 2]), name="global_avg_pool")(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    return models.Model(inputs=x_in, outputs=out, name="STGCN_12KP")