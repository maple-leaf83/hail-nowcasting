"""
networks.py
-----------
Defines the MLP neural network architecture used for severe hail nowcasting.

The model uses a shared trunk of fully-connected layers to learn common
representations across three forecast horizons, with small task-specific
head layers before each sigmoid output to allow per-target specialisation.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import ops
from tensorflow.keras import layers


def gen_mlp_network(input_shape, num_filters, dropout, head_units=64):
    """
    Build a multi-output MLP for simultaneous hail probability prediction
    at three forecast lead times.

    Architecture:
        Shared trunk: Dense(num_filters[0], relu)
                      → [BatchNorm → Dropout → Dense(num_filters[i], relu)] × (len(num_filters)-1)
        Per-task heads (×3): BatchNorm → Dropout → Dense(head_units, relu) → Dense(1, sigmoid)

    The shared trunk learns feature representations common to all three targets.
    Each task-specific head then adapts those representations to its own target
    distribution, allowing the model to specialise without fully independent
    networks.

    Args:
        input_shape (tuple): Shape of the input feature vector, e.g. (37,).
        num_filters (list):  Number of units in each trunk layer, e.g. [512, 512, 512, 512, 512].
        dropout (float):     Dropout rate applied in each trunk and head block, e.g. 0.1.
        head_units (int):    Number of units in the task-specific dense layer of each head.
                             Default: 64.

    Returns:
        tf.keras.Model: Compiled-ready multi-output model with three sigmoid outputs:
                        [y_train_1 (0-30 min), y_train_2 (15-45 min), y_train_3 (30-60 min)].
    """
    inputs = Input(shape=(input_shape))

    # Shared trunk: first layer has no preceding BatchNorm/Dropout
    x = Dense(num_filters[0], activation='relu', name='dense0')(inputs)

    for i in range(1, len(num_filters)):
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(num_filters[i], activation='relu',name='dense'+str(i))(x)

    # Define four separate output layers
    # NOTE: outputs used to branch directly off the shared trunk `x` with no
    # task-specific capacity. Replaced below with small per-head blocks so
    # each of the 3 targets can specialize. Kept here, commented, for reference.
    # output1 = Dense(1, activation='sigmoid', name='y_train_1')(x)
    # output2 = Dense(1, activation='sigmoid', name='y_train_2')(x)
    # output3 = Dense(1, activation='sigmoid', name='y_train_3')(x)
    # output4 = Dense(1, activation='sigmoid', name='y_train_4')(x)

    def make_head(shared, name):
        """ Small task-specific block before each sigmoid output, so each
        head gets some of its own capacity instead of sharing the entire
        trunk. """
        h = BatchNormalization()(shared)
        h = Dropout(dropout)(h)
        h = Dense(head_units, activation='relu', name=name + '_dense')(h)
        return Dense(1, activation='sigmoid', name=name)(h)

    output1 = make_head(x, 'y_train_1')
    output2 = make_head(x, 'y_train_2')
    output3 = make_head(x, 'y_train_3')

        # Create the model
    model = Model(inputs=inputs, outputs=[output1, output2, output3])

    return model


