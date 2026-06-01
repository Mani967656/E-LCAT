"""
Put your custom Keras layers here so the `.h5` model can load.

Your error says the model needs: ColorAwareAttention

If you have the training code, copy the exact layer class implementation into
this file, then expose it via `get_custom_objects()`.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential


class ColorAwareAttention(Layer):
    """Color-Aware Attention layer."""

    def __init__(self, **kwargs):
        super(ColorAwareAttention, self).__init__(**kwargs)
        self.avg_pool = GlobalAveragePooling2D()

    def call(self, inputs):
        # inputs: (batch, H, W, C)
        mean_color = self.avg_pool(inputs)  # (batch, C)
        mean_color_spatial = tf.expand_dims(tf.expand_dims(mean_color, 1), 1)  # (batch,1,1,C)
        squared_diff = tf.math.square(inputs - mean_color_spatial)
        sum_squared_diff = tf.reduce_sum(squared_diff, axis=-1, keepdims=True)
        distance = tf.math.sqrt(sum_squared_diff + 1e-6)
        attention_map = tf.nn.sigmoid(distance)
        attended_features = inputs * attention_map
        output = inputs + attended_features
        return output

    def get_config(self):
        return super(ColorAwareAttention, self).get_config()


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        if self.embed_dim % max(1, self.num_heads) != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = self.embed_dim // max(1, self.num_heads)

        # Dense projections (match legacy weight shapes like (embed_dim, embed_dim))
        self.q = Dense(self.embed_dim, use_bias=True)
        self.k = Dense(self.embed_dim, use_bias=True)
        self.v = Dense(self.embed_dim, use_bias=True)
        self.proj = Dense(self.embed_dim, use_bias=True)
        self.drop = Dropout(self.dropout)

    def call(self, inputs, training=False):
        # inputs: (batch, tokens, embed_dim)
        b = tf.shape(inputs)[0]
        t = tf.shape(inputs)[1]

        q = self.q(inputs)  # (b, t, embed_dim)
        k = self.k(inputs)
        v = self.v(inputs)

        # (b, heads, t, head_dim)
        q = tf.reshape(q, (b, t, self.num_heads, self.head_dim))
        k = tf.reshape(k, (b, t, self.num_heads, self.head_dim))
        v = tf.reshape(v, (b, t, self.num_heads, self.head_dim))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        scale = tf.cast(self.head_dim, tf.float32) ** -0.5
        attn_logits = tf.matmul(q, k, transpose_b=True) * scale  # (b, heads, t, t)
        attn = tf.nn.softmax(attn_logits, axis=-1)
        attn = self.drop(attn, training=training)
        out = tf.matmul(attn, v)  # (b, heads, t, head_dim)

        out = tf.transpose(out, (0, 2, 1, 3))  # (b, t, heads, head_dim)
        out = tf.reshape(out, (b, t, self.embed_dim))
        out = self.proj(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return cfg


class LCATBlock(Layer):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, drop_rate=0.0, **kwargs):
        super(LCATBlock, self).__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_rate = float(drop_rate)

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = Sequential(
            [
                Dense(mlp_hidden_dim, activation="gelu"),
                Dropout(self.drop_rate),
                Dense(self.embed_dim),
                Dropout(self.drop_rate),
            ]
        )

    def call(self, inputs, training=False):
        x1 = self.norm1(inputs)
        attention_output = self.attn(x1, training=training)
        x2 = inputs + attention_output
        x3 = self.norm2(x2)
        mlp_output = self.mlp(x3, training=training)
        output = x2 + mlp_output
        return output

    def get_config(self):
        cfg = super(LCATBlock, self).get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
            }
        )
        return cfg


class PatchMerging(Layer):
    def __init__(self, new_dim, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.new_dim = int(new_dim)
        self.conv_down = Conv2D(self.new_dim, kernel_size=2, strides=2, padding="valid")

    def call(self, inputs):
        # inputs shape: (batch, num_patches, C)
        shape = tf.shape(inputs)
        num_patches = shape[1]
        current_dim = shape[2]
        h = w = tf.cast(tf.round(tf.sqrt(tf.cast(num_patches, tf.float32))), tf.int32)
        x = tf.reshape(inputs, (-1, h, w, current_dim))
        x = self.conv_down(x)
        h2 = tf.shape(x)[1]
        w2 = tf.shape(x)[2]
        output = tf.reshape(x, (-1, h2 * w2, self.new_dim))
        return output

    def get_config(self):
        cfg = super(PatchMerging, self).get_config()
        cfg.update({"new_dim": self.new_dim})
        return cfg


def get_custom_objects() -> dict[str, Any]:
    """
    Return custom_objects dict for tf.keras.models.load_model(...).

    Example (after you paste your layer class here):

        from .custom_layers import ColorAwareAttention

        def get_custom_objects():
            return {"ColorAwareAttention": ColorAwareAttention}
    """

    return {
        "ColorAwareAttention": ColorAwareAttention,
        "MultiHeadSelfAttention": MultiHeadSelfAttention,
        "LCATBlock": LCATBlock,
        "PatchMerging": PatchMerging,
    }

