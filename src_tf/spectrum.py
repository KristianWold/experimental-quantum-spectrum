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


def complex_spacing_ratio(spectrum, verbose=False, log=True):
    d = len(spectrum)
    spectrum = np.array(spectrum)[:, 0]
    if log:
        spectrum = np.log(spectrum)

    z_list = []
    if verbose:
        decorator = tqdm
    else:
        decorator = lambda x: x

    for i in decorator(range(d)):
        idx_NN = i
        dist_NN = float("inf")

        idx_NNN = i
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN
                    idx_NNN = idx_NN

                    dist_NN = dist
                    idx_NN = j

                if (dist > dist_NN) and (dist < dist_NNN):
                    dist_NNN = dist
                    idx_NNN = j

            z = (spectrum[idx_NN] - spectrum[i]) / (spectrum[idx_NNN] - spectrum[i])
        z_list.append(z)

    return np.array(z_list)


def spacing_ratio(spectrum):
    d = len(spectrum)
    z_list = []
    for i in tqdm(range(d)):
        idx_NN = i
        dist_NN = float("inf")

        idx_NNN = i
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.angle(spectrum[i] - spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN
                    idx_NNN = idx_NN

                    dist_NN = dist
                    idx_NN = j

                if (dist > dist_NN) and (dist < dist_NNN):
                    dist_NNN = dist
                    idx_NNN = j

        z = np.angle(spectrum[i] - spectrum[idx_NN]) / np.angle(
            spectrum[i] - spectrum[idx_NNN]
        )
        z_list.append(z)


def distance_spacing_ratio(spectrum, verbose=False):
    d = len(spectrum)
    spectrum = np.array(spectrum)[:, 0]
    z_list = []
    if verbose:
        decorator = tqdm
    else:
        decorator = lambda x: x
    log_spectrum = np.log(spectrum)
    s = mean_spacing(log_spectrum)
    rho = unfolding(log_spectrum, 4.5 * s)

    for i in decorator(range(d)):
        dist_NN = float("inf")
        dist_NNN = float("inf")

        for j in range(d):
            if j != i:
                dist = np.abs(log_spectrum[i] - log_spectrum[j])
                if dist < dist_NN:
                    dist_NNN = dist_NN

                    dist_NN = dist

                if (dist > dist_NN) and (dist < dist_NNN):
                    dist_NNN = dist

        z = dist_NNN / dist_NN
        z = z * np.sqrt(rho[i])
        z_list.append(z)

    return np.array(z_list)


def unfolding(spectrum, sigma):
    N = spectrum.shape[0]
    spectrum = np.array(spectrum)
    diff = np.abs(spectrum.reshape(-1, 1) - spectrum.reshape(1, -1))
    expo = -1 / (2 * sigma**2) * diff**2
    rho = 1 / (2 * np.pi * sigma**2 * N) * np.sum(np.exp(expo), axis=1)
    return rho


def mean_spacing(spectrum):
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


def hopkins_statistic(spectrum, split=10):
    angles = np.angle(spectrum)
    N = len(angles)
    m = N//split
    
    idx = np.random.choice(N, m, replace=False)
    X = angles[idx]
    Y = np.random.uniform(0, 2*np.pi, m)
    distance_X = np.abs(X.reshape(-1, 1) - angles.reshape(1, -1))
    distance_X[distance_X == 0] = np.inf
    u = np.min(distance_X, axis=1)

    distance_Y = np.abs(Y.reshape(-1, 1) - angles.reshape(1, -1))
    w = np.min(distance_Y, axis=1)
    
    u_sum = np.sum(u)
    w_sum = np.sum(w)

    hs = u_sum / (u_sum + w_sum)

    return hs, u_sum, w_sum
    
