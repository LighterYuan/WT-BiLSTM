from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from topology_attention import TopologyAttention


def _compile(model: tf.keras.Model, cfg: Dict[str, Any]) -> tf.keras.Model:
    lr = float(cfg["training"].get("learning_rate", 0.001))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_model(
    model_name: str,
    input_shape: Tuple[int, int],
    cfg: Dict[str, Any],
    topology_shape: Optional[Tuple[int, int]] = None,
) -> tf.keras.Model:
    name = model_name.upper().replace("_", "-")
    mcfg = cfg["model"]
    units = int(mcfg.get("units", 64))
    dense_units = int(mcfg.get("dense_units", 32))
    dropout = float(mcfg.get("dropout", 0.2))

    if name == "GRU":
        inp = layers.Input(shape=input_shape, name="traffic_window")
        x = layers.GRU(units, name="gru")(inp)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, name="prediction")(x)
        return _compile(models.Model(inp, out, name="GRU"), cfg)

    if name in {"LSTM", "WT-LSTM"}:
        inp = layers.Input(shape=input_shape, name="traffic_window")
        x = layers.LSTM(units, name="lstm")(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        out = layers.Dense(1, name="prediction")(x)
        return _compile(models.Model(inp, out, name=name.replace("-", "_")), cfg)

    if name in {"BILSTM", "WT-BILSTM"}:
        inp = layers.Input(shape=input_shape, name="traffic_window")
        x = layers.Bidirectional(layers.LSTM(units), name="bilstm")(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        out = layers.Dense(1, name="prediction")(x)
        return _compile(models.Model(inp, out, name=name.replace("-", "_")), cfg)

    if name == "CNN-LSTM":
        inp = layers.Input(shape=input_shape, name="traffic_window")
        x = layers.Conv1D(
            filters=int(mcfg.get("conv_filters", 32)),
            kernel_size=int(mcfg.get("conv_kernel_size", 3)),
            padding="causal",
            activation="relu",
            name="causal_conv1d",
        )(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(units, name="lstm_after_cnn")(x)
        x = layers.Dense(dense_units, activation="relu")(x)
        out = layers.Dense(1, name="prediction")(x)
        return _compile(models.Model(inp, out, name="CNN_LSTM"), cfg)

    if name == "WT-TBILSTM":
        if topology_shape is None:
            raise ValueError("WT-TBiLSTM requires topology_shape.")
        seq_in = layers.Input(shape=input_shape, name="traffic_window")
        topo_in = layers.Input(shape=topology_shape, name="topology_features")
        h = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True),
            name="bilstm_sequence_encoder",
        )(seq_in)
        h = layers.Dropout(dropout)(h)
        context = TopologyAttention(
            attention_units=int(mcfg.get("attention_units", 32)),
            name="topology_attention",
        )([h, topo_in])
        x = layers.Dense(dense_units, activation="relu")(context)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(1, name="prediction")(x)
        return _compile(models.Model([seq_in, topo_in], out, name="WT_TBiLSTM"), cfg)

    raise ValueError(f"Unsupported model name: {model_name}")


def model_input_summary(model_name: str) -> str:
    name = model_name.upper().replace("_", "-")
    if name == "WT-TBILSTM":
        return "traffic_window + topology_features"
    return "traffic_window"
