import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops


def huber(y_true, y_pred):
    """Computes Huber loss value.

    For each value x in `error = y_true - y_pred`:

    ```
    loss = 0.5 * x^2        if |x| <= d
    loss = |x| - 1/2        if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.

    Returns:
      Tensor with one scalar loss entry per sample.
    """
    y_pred = math_ops.cast(y_pred, dtype=K.floatx())
    y_true = math_ops.cast(y_true, dtype=K.floatx())
    const = ops.convert_to_tensor_v2(0.5, dtype=y_pred.dtype)
    error = math_ops.subtract(y_pred, y_true)
    abs_error = math_ops.subtract(math_ops.abs(error), const)
    quadratic = math_ops.multiply(math_ops.multiply(error, error), const)

    return \
        K.mean(
            tf.where(
                quadratic >= abs_error,
                abs_error,
                quadratic
            )
        )
