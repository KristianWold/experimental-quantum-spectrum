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


def channel_spectrum(input, use_coords=True, keep_real=True, keep_unity=True, tol=1e-4):
    eig, _ = tf.linalg.eig(reshuffle(input.choi))
    eig = tf.expand_dims(eig, axis=1)

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


def mean_spacing(spectrum):
    if len(spectrum.shape) == 2:
        spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    d = len(spectrum)
    ms_list = []
    for i in range(d):
        idx_NN = i
        dist_NN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NN = dist

        ms_list.append(dist_NN)

    return np.mean(ms_list)


def coat_spectrum(spectrum, sigma=0.1, grid_size=100):
    """Coat each eigenvalue with a Gaussian distribution."""

    spectrum = np.real(spectrum)
    x_grid = np.linspace(-1, 1, grid_size)
    y_grid = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    rho = 0
    for eig in spectrum:
        rho += np.exp(-((X - eig[0]) ** 2 + (Y - eig[1]) ** 2) / (2 * sigma**2))

    return rho



