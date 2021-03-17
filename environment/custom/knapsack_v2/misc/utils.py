import tensorflow as tf
import numpy as np

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    # Replace math.floor with np.floor
    return np.floor(n*multiplier + 0.5) / multiplier

def compute_remaining_resources(bins, items, decimal_precision = 2):
    
    # bins_current_load = bins[:, 1]
    # items_weight = items[:, 0]
    updated_bin = bins.copy()
    
    updated_bin[:, 1] = updated_bin[:, 1] + items[:, 0]

    updated_bin = round_half_up(updated_bin, decimal_precision)

    return updated_bin

def bins_eos_checker(bins, EOS_SYMBOL, num_features, result_dtype = 'int32'):
    # Check if selected bins are EOS
    # Marked as 1 = EOS node
    # Marked as 0 = not a EOS node
    is_eos_bin = tf.cast(tf.equal(bins, EOS_SYMBOL), dtype = result_dtype)
    # if all elements are equal to EOS code
    # the result should be equal to the number of features
    is_eos_bin = tf.reduce_sum(is_eos_bin, axis=-1)
    is_eos_bin = tf.cast(tf.equal(is_eos_bin, num_features), dtype = result_dtype)

    return is_eos_bin