from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops, array_ops


def clipped_mse(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(K.clip(math_ops.squared_difference(y_pred, y_true), -1, 1), axis=-1)


def huber(y_true, y_pred, delta=1.0):
  """Computes Huber loss value.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  y_pred = math_ops.cast(y_pred, dtype=K.floatx())
  y_true = math_ops.cast(y_true, dtype=K.floatx())
  delta = math_ops.cast(delta, dtype=K.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  half = ops.convert_to_tensor_v2_with_dispatch(0.5, dtype=abs_error.dtype)
  return K.mean(
      array_ops.where_v2(
          abs_error <= delta, half * math_ops.pow(error, 2),
          abs_error - half),
      axis=-1)