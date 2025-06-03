import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import tensorflow as tf

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from quantum_tools import *
from utils import *
from experimental import *
from set_precision import *
from quantum_channel import *
from spectrum import *


# Loss functions
#######################################################
def state_fidelity_loss(channel, input, target):
    """Negative quantum state fidelity"""
    state = input
    output = channel.apply_map(state)
    loss = -state_fidelity(output, target)
    return loss


def expectation_value_loss(channel, input, target):
    """MSE Loss over measured expectation values"""
    state, observable = input
    state = channel.apply_map(state)
    output = expectation_value(state, observable)
    loss = tf.abs(output - target) ** 2
    return loss


class ProbabilityMSE:
    """MSE loss over measured computational basis probabilities"""

    def __call__(self, channel, input, target):
        N = target.shape[0]
        d = channel.spam.d
        if isinstance(input, list):
            U_prep, U_basis = input
        else:
            U_prep = input
            U_basis = None

        state = tf.repeat(tf.expand_dims(channel.spam.init.init, axis=0), N, axis=0)
        state = apply_unitary(state, U_prep)
        state = channel.apply_channel(state)
        output = measurement(state, U_basis, channel.spam.povm.povm)
        loss = d**2 * tf.math.reduce_mean((output - target) ** 2)

        return loss


class ProbabilityRValue:
    """MSE loss over measured computational basis probabilities"""

    def __call__(self, channel, input, target):
        N = target.shape[0]
        d = channel.spam.d

        if isinstance(input, list):
            U_prep, U_basis = input
        else:
            U_prep = input
            U_basis = None

        state = tf.repeat(tf.expand_dims(channel.spam.init.init, axis=0), N, axis=0)
        state = apply_unitary(state, U_prep)
        state = channel.apply_channel(state)
        output = measurement(state, U_basis, channel.spam.povm.povm)
        loss = 1 - (
            tf.math.reduce_mean(tf.abs(output - target) ** 2)
            / tf.math.reduce_std(target) ** 2
        )

        return loss


class KLDiv:
    """KL-Divergence over measured computational basis probabilities"""

    def __call__(self, channel, input, target, normalize=True):
        N = target.shape[0]
        U_prep, U_basis = input

        state = tf.repeat(tf.expand_dims(channel.spam.init.init, axis=0), N, axis=0)
        state = apply_unitary(state, U_prep)
        state = channel.apply_channel(state)

        mask = tf.math.real(target) > 1e-15

        output = measurement(state, U_basis, channel.spam.povm.povm)
        loss = tf.math.reduce_sum(
            target[mask] * tf.math.log(target[mask] / output[mask])
        )

        if normalize:
            loss /= N
        return loss



def channel_fidelity_loss(channel, input, target):
    """Negative channel fidelity between quantum channels"""
    channel_target = target[0]
    loss = -channel_fidelity(channel, channel_target)
    return loss


def channel_mse_loss(channel, input, target):
    """Elementwise MSE loss between choi matrices"""
    channel_target = target[0]
    choi_model = channel.choi
    choi_target = channel_target.choi

    loss = tf.math.reduce_sum(tf.abs(choi_model - choi_target) ** 2)
    return loss


class SpectrumDistance:
    """Distance measure between spectra"""

    def __init__(self, sigma=0.1, k=1000, mode="density", remove_shift=True):
        self.sigma = sigma
        self.sigma_ = sigma
        self.k = k
        self.remove_shift = remove_shift
        self.mode = mode
        self.t = 0

    def __call__(self, channel, input, target):
        spectrum_target = target[0]
        spectrum_model = channel_spectrum(channel)

        if self.mode == "density":
            loss = self.overlap(spectrum_model, spectrum_model)
            loss += -2 * self.overlap(spectrum_model, spectrum_target)
            if not self.remove_shift:
                loss += self.overlap(spectrum_target, spectrum_target)

        if self.mode == "pairwise":
            connections = self.greedy_pair_distance(spectrum_model, spectrum_target)
            loss = self.pair_distance(spectrum_model, spectrum_target, connections)

        self.t += 1
        if self.k is not None:
            self.sigma = np.sqrt(self.k) * self.sigma_ / np.sqrt(self.k + self.t)
        else:
            self.sigma = self.sigma_

        return loss

    def overlap(self, spectrum_a, spectrum_b):
        aa = tf.math.reduce_sum(spectrum_a * spectrum_a, axis=1, keepdims=True)
        bb = tf.math.reduce_sum(spectrum_b * spectrum_b, axis=1, keepdims=True)
        ab = tf.matmul(spectrum_a, spectrum_b, adjoint_b=True)

        expo = aa - 2 * ab + tf.transpose(bb)
        sum = (
            1
            / np.sqrt(self.sigma)
            * tf.math.reduce_mean(tf.math.exp(-expo / self.sigma**2))
        )

        return sum

    def greedy_pair_distance(self, spectrum_a, spectrum_b):
        connections = []
        not_connected = len(spectrum_a) * [True]

        spectrum_a = tf.stop_gradient(spectrum_a)
        spectrum_b = tf.stop_gradient(spectrum_b)

        aa = tf.math.reduce_sum(spectrum_a * spectrum_a, axis=1, keepdims=True)
        bb = tf.math.reduce_sum(spectrum_b * spectrum_b, axis=1, keepdims=True)
        ab = tf.matmul(spectrum_a, spectrum_b, adjoint_b=True)

        dist = aa - 2 * ab + tf.transpose(bb)

        for i, a in enumerate(spectrum_a):
            min_dist = float("inf")
            idx = 0
            for j, b in enumerate(spectrum_b):
                dist = tf.abs((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                if (dist < min_dist) and not_connected[j]:
                    min_dist = dist
                    idx = j

            not_connected[idx] = False
            connections.append(idx)

        return connections

    def pair_distance(self, spectrum_a, spectrum_b, connections):
        distance = 0
        for i, idx in enumerate(connections):
            distance += (spectrum_a[i][0] - spectrum_b[idx][0]) ** 2 + (
                spectrum_a[i][1] - spectrum_b[idx][1]
            ) ** 2

        return distance


class AnnulusDistance:
    def __call__(self, channel, input, target):
        spectrum_target = target[0]
        spectrum_model = channel_spectrum(channel, keep_unity=False)

        r_mean1, r_std1, _, _ = self.spectrum_to_momenta(spectrum_model)
        r_mean2, r_std2, _, _ = self.spectrum_to_momenta(spectrum_target)

        distance = (
            tf.math.abs(r_mean1 - r_mean2)
            + tf.math.abs(r_std1 - r_std2)

        )

        return distance

    def spectrum_to_momenta(self, spectrum):
        radial = spectrum_to_radial(spectrum)
        r_mean = tf.math.reduce_mean(radial)
        r_std = tf.math.reduce_std(radial)

        return r_mean, r_std, None, None







