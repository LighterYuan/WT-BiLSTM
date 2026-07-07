from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def _compile(model: tf.keras.Model, cfg: Dict[str, Any]) -> tf.keras.Model:
    lr = float(cfg["training"].get("learning_rate", 0.001))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


class PositionalEncoding(layers.Layer):
    """Trainable positional embedding for short univariate traffic windows."""

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.pos_embedding = layers.Embedding(input_dim=self.max_len, output_dim=self.d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return x + self.pos_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


def _transformer_encoder(x, d_model: int, num_heads: int, ff_dim: int, dropout: float, name_prefix: str):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=max(1, d_model // num_heads), name=f"{name_prefix}_mha")(x, x)
    attn = layers.Dropout(dropout, name=f"{name_prefix}_attn_dropout")(attn)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_attn_norm")(x + attn)

    ff = layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ffn_1")(x)
    ff = layers.Dropout(dropout, name=f"{name_prefix}_ffn_dropout")(ff)
    ff = layers.Dense(d_model, name=f"{name_prefix}_ffn_2")(ff)
    return layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ffn_norm")(x + ff)


def build_tcn(input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    mcfg = cfg["model"]
    dropout = float(mcfg.get("dropout", 0.2))
    filters = int(mcfg.get("tcn_filters", mcfg.get("units", 64)))
    kernel_size = int(mcfg.get("tcn_kernel_size", 3))

    inp = layers.Input(shape=input_shape, name="traffic_window")
    x = inp
    for i, dilation in enumerate([1, 2, 4, 8]):
        residual = x
        x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu", name=f"tcn_conv_{i}_a")(x)
        x = layers.Dropout(dropout, name=f"tcn_dropout_{i}_a")(x)
        x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu", name=f"tcn_conv_{i}_b")(x)
        if residual.shape[-1] != filters:
            residual = layers.Conv1D(filters, 1, padding="same", name=f"tcn_residual_{i}")(residual)
        x = layers.Add(name=f"tcn_add_{i}")([x, residual])
        x = layers.LayerNormalization(name=f"tcn_norm_{i}")(x)
    x = layers.Lambda(lambda z: z[:, -1, :], name="last_time_step")(x)
    x = layers.Dense(int(mcfg.get("dense_units", 32)), activation="relu", name="dense_repr")(x)
    out = layers.Dense(1, name="prediction")(x)
    return _compile(models.Model(inp, out, name="TCN"), cfg)


def build_transformer(input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    mcfg = cfg["model"]
    d_model = int(mcfg.get("transformer_d_model", 64))
    heads = int(mcfg.get("transformer_heads", 4))
    ff_dim = int(mcfg.get("transformer_ff_dim", 128))
    blocks = int(mcfg.get("transformer_blocks", 2))
    dropout = float(mcfg.get("dropout", 0.2))

    inp = layers.Input(shape=input_shape, name="traffic_window")
    x = layers.Dense(d_model, name="input_projection")(inp)
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model, name="positional_encoding")(x)
    for i in range(blocks):
        x = _transformer_encoder(x, d_model, heads, ff_dim, dropout, name_prefix=f"encoder_{i}")
    x = layers.GlobalAveragePooling1D(name="temporal_pooling")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(int(mcfg.get("dense_units", 32)), activation="relu", name="dense_repr")(x)
    out = layers.Dense(1, name="prediction")(x)
    return _compile(models.Model(inp, out, name="Transformer"), cfg)


def build_dlinear(input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    # DLinear-style decomposition baseline: moving-average trend + residual seasonal linear heads.
    inp = layers.Input(shape=input_shape, name="traffic_window")
    pool_size = min(3, input_shape[0])
    trend = layers.AveragePooling1D(pool_size=pool_size, strides=1, padding="same", name="moving_average_trend")(inp)
    seasonal = layers.Subtract(name="seasonal_residual")([inp, trend])
    trend_flat = layers.Flatten(name="trend_flatten")(trend)
    seasonal_flat = layers.Flatten(name="seasonal_flatten")(seasonal)
    trend_out = layers.Dense(1, use_bias=True, name="trend_linear")(trend_flat)
    seasonal_out = layers.Dense(1, use_bias=True, name="seasonal_linear")(seasonal_flat)
    out = layers.Add(name="prediction")([trend_out, seasonal_out])
    return _compile(models.Model(inp, out, name="DLinear"), cfg)


def build_nlinear(input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    # NLinear-style normalization by the last observed value in the input window.
    inp = layers.Input(shape=input_shape, name="traffic_window")
    last = layers.Lambda(lambda z: z[:, -1:, :], name="last_observation")(inp)
    centered = layers.Subtract(name="centered_window")([inp, last])
    x = layers.Flatten(name="flatten_centered")(centered)
    delta = layers.Dense(1, name="linear_delta")(x)
    last_scalar = layers.Lambda(lambda z: z[:, 0, :], name="last_scalar")(last)
    out = layers.Add(name="prediction")([delta, last_scalar])
    return _compile(models.Model(inp, out, name="NLinear"), cfg)


def _frame_patches(x, patch_len: int, stride: int):
    # x: (batch, time, 1) -> (batch, num_patches, patch_len)
    x = tf.squeeze(x, axis=-1)
    return tf.signal.frame(x, frame_length=patch_len, frame_step=stride, pad_end=False)


def build_patchtst(input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    # Lightweight PatchTST-style baseline for short univariate windows.
    mcfg = cfg["model"]
    d_model = int(mcfg.get("transformer_d_model", 64))
    heads = int(mcfg.get("transformer_heads", 4))
    ff_dim = int(mcfg.get("transformer_ff_dim", 128))
    blocks = int(mcfg.get("transformer_blocks", 2))
    dropout = float(mcfg.get("dropout", 0.2))
    patch_len = min(int(mcfg.get("patch_len", 4)), input_shape[0])
    stride = max(1, int(mcfg.get("patch_stride", 2)))
    num_patches = max(1, 1 + (input_shape[0] - patch_len) // stride)

    inp = layers.Input(shape=input_shape, name="traffic_window")
    patches = layers.Lambda(lambda z: _frame_patches(z, patch_len, stride), name="patching")(inp)
    x = layers.Dense(d_model, name="patch_projection")(patches)
    x = PositionalEncoding(max_len=num_patches, d_model=d_model, name="patch_positional_encoding")(x)
    for i in range(blocks):
        x = _transformer_encoder(x, d_model, heads, ff_dim, dropout, name_prefix=f"patch_encoder_{i}")
    x = layers.GlobalAveragePooling1D(name="patch_pooling")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, name="prediction")(x)
    return _compile(models.Model(inp, out, name="PatchTST_style"), cfg)


def build_major_revision_model(model_name: str, input_shape: Tuple[int, int], cfg: Dict[str, Any]) -> tf.keras.Model:
    name = model_name.upper().replace("_", "-")
    if name == "TCN":
        return build_tcn(input_shape, cfg)
    if name == "TRANSFORMER":
        return build_transformer(input_shape, cfg)
    if name == "DLINEAR":
        return build_dlinear(input_shape, cfg)
    if name == "NLINEAR":
        return build_nlinear(input_shape, cfg)
    if name in {"PATCHTST", "PATCHTST-STYLE", "PATCHTST_STYLE"}:
        return build_patchtst(input_shape, cfg)
    raise ValueError(f"Unsupported major-revision model: {model_name}")
