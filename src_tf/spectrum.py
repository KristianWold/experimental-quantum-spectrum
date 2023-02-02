import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator, random_unitary
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from quantum_tools import *
from utils import *
from set_precision import *
from quantum_channel import *


def channel_spectrum(input, use_coords=True, keep_real=True, keep_unity=True, tol=1e-4):
    if isinstance(input, Channel):
        eig, _ = tf.linalg.eig(reshuffle(input.choi))
        eig = tf.expand_dims(eig, axis=1)
    else:
        eig = input

    if not keep_real:
        mask = tf.abs(tf.math.imag(eig)) > tol
        eig = tf.expand_dims(eig[mask], axis=1)

    if not keep_unity:
        mask = tf.abs(tf.math.real(eig) - 1) > tol
        eig = tf.expand_dims(eig[mask], axis=1)

    if use_coords:
        x = tf.cast(tf.math.real(eig), dtype=precision)
        y = tf.cast(tf.math.imag(eig), dtype=precision)
        eig = tf.concat([x, y], axis=1)

    return eig


def choi_spectrum(channel):
    eig, _ = tf.linalg.eig(channel.choi)
    eig = tf.expand_dims(eig, axis=1)

    return eig


def normalize_spectrum(spectrum, scale=1):
    spectrum = spectrum.numpy()
    idx = np.argmax(np.linalg.norm(spectrum, axis=1))
    spectrum[idx] = (0, 0)

    max = np.max(np.linalg.norm(spectrum, axis=1))
    spectrum = scale / max * spectrum

    spectrum[idx] = (1, 0)
    spectrum = tf.cast(tf.convert_to_tensor(spectrum), dtype=precision)

    return spectrum
