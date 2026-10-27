"""Only FastOBQ geometry from RCO model_utils.py is retained here.

Source: IST-DASLab/RCO 9a1e09c07d468109cbe60a1b87d5036034a79d10,
Apache-2.0. Local change: math.prod replaces numpy.prod; unrelated model
loading and mutation helpers are intentionally omitted.
"""
import math


def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], math.prod(layer.weight.shape[1:])
