# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Shows how to use tfe.gradients_function(f) to
# compute the gradient of a function
#
import tensorflow as tf
print("Your TF version is", tf.__version__)

tf.enable_eager_execution()
tfe = tf.contrib.eager # Shorthand for some symbols


def f(x, y):
  output = 1
  # Must use range(int(y)) instead of range(y) in Python 3 when
  # using TensorFlow 1.10 and earlier. Can use range(y) in 1.11+
  for i in range(int(y)):
    output = tf.multiply(output, x)
  return output

def g(x, y):
  # Return the gradient of `f` with respect to it's first parameter
  return tfe.gradients_function(f)(x, y)[0]

assert f(3.0, 2).numpy() == 9.0   # f(x, 2) is essentially x * x
assert g(3.0, 2).numpy() == 6.0   # And its gradient will be 2 * x
assert f(4.0, 3).numpy() == 64.0  # f(x, 3) is essentially x * x * x
assert g(4.0, 3).numpy() == 48.0  # And its gradient will be 3 * x * x
