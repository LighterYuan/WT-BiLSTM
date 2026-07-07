from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="WTTBiLSTM")
class TopologyAttention(tf.keras.layers.Layer):
    """Topology-aware temporal attention.

    Inputs:
        hidden_states: shape (batch, time_steps, hidden_dim)
        topology_features: shape (batch, time_steps, topo_dim)
    Output:
        context: shape (batch, hidden_dim)
    """

    def __init__(self, attention_units: int = 32, return_attention: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.attention_units = int(attention_units)
        self.return_attention = bool(return_attention)
        self.proj = tf.keras.layers.Dense(self.attention_units, activation="tanh")
        self.score = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        hidden_states, topology_features = inputs
        z = tf.concat([hidden_states, topology_features], axis=-1)
        e = self.score(self.proj(z))
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * hidden_states, axis=1)
        if self.return_attention:
            return context, tf.squeeze(alpha, axis=-1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({
            "attention_units": self.attention_units,
            "return_attention": self.return_attention,
        })
        return config
