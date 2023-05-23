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

    def __call__(self, channel, input, target):
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

        return loss


class LogLikelihood:
    def __call__(self, channel, input, target):
        N = target.shape[0]
        d = channel.spam.init.shape[0]
        U_prep, U_basis = input

        state = tf.repeat(tf.expand_dims(channel.spam.init, axis=0), N, axis=0)
        state = apply_unitary(state, U_prep)
        state = channel.apply_channel(state)
        output = measurement(state, U_basis, channel.spam.povm)
        loss = tf.math.reduce_sum(target * tf.math.log(output))

        return loss


class RankMSE:
    """MSE loss on effective kraus rank of channel"""

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, channel, input, target):
        rank_target = target
        loss = self.weight * (effective_rank(channel) - rank_target) ** 2

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

    def __init__(self, sigma=0.1, k=1000):
        self.sigma = sigma
        self.sigma_ = sigma
        self.k = k
        self.mode = "density"
        self.t = 0

    def __call__(self, channel, input, target):
        spectrum_target = input[0]
        spectrum_model = channel_spectrum(channel)

        if self.mode == "density":
            loss = self.overlap(spectrum_model, spectrum_model)
            loss += -2 * self.overlap(spectrum_model, spectrum_target)
            # loss += self.overlap(spectrum_target, spectrum_target)

        if self.mode == "pairwise":
            connections = greedy_pair_distance(spectrum_model, spectrum_target)
            loss = pair_distance(spectrum_model, spectrum_target, connections)

        self.t += 1
        self.sigma = np.sqrt(self.k) * self.sigma_ / np.sqrt(self.k + self.t)

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

    """
    def greedy_pair_distance(self, spectrum_a, spectrum_b):
        connections = []
        not_connected = len(spectrum_a) * [True]

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
    """

    def greedy_pair_distance(self, spectrum_a, spectrum_b):
        connections = []
        not_connected = len(spectrum_a) * [True]

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
        spectrum_model = channel_spectrum(channel, use_coords=True, keep_unity=False)

        r_mean1, r_std1, a_mean1, a_std1 = self.spectrum_to_momenta(spectrum_model)
        r_mean2, r_std2, a_mean2, a_std2 = self.spectrum_to_momenta(spectrum_target)

        distance = (
            tf.math.abs(r_mean1 - r_mean2)
            + tf.math.abs(r_std1 - r_std2)
            # + tf.math.abs(a_mean1 - a_mean2)
            #    + tf.math.abs(a_std1 - a_std2)
        )

        return distance

    def spectrum_to_momenta(self, spectrum):
        radial = spectrum_to_radial(spectrum)
        r_mean = tf.math.reduce_mean(radial)
        r_std = tf.math.reduce_std(radial)

        #   angular = spectrum_to_angular(spectrum) / (2 * np.pi)
        #   a_mean = tf.math.reduce_mean(angular)
        #   a_std = tf.math.reduce_std(angular)

        return r_mean, r_std, None, None


# Regularizers
#######################################################


class RankShrink:
    """Penalize effective kraus rank of channel"""

    def __init__(self, inflate=False, weight=1):
        self.weight = weight

        if inflate:
            self.sign = -1
        else:
            self.sign = 1

    def __call__(self, channel, input, target):
        loss = self.sign * self.weight * effective_rank(channel)

        return tf.math.abs(loss)


class AttractionShrink:
    """Penalize effective kraus rank of channel"""

    def __init__(self, inflate=False, weight=1, N=1000):
        self.weight = weight
        self.N = N

        if inflate:
            self.sign = -1
        else:
            self.sign = 1

    def __call__(self, channel, input, target):
        loss = (
            self.sign
            * self.weight
            * tf.cast(attraction(channel, N=self.N), dtype=precision)
        )

        return loss


# Adverserial Loss
#######################################################

# class AttractionRankTradeoff:
#    """Optimize towards channel with high(low) effective rank,
#       and low(high) attraction"""
#
#    def __init__(self, inflate=False, weight = 1):
#        if inflate:
#            self.sign = -1
#        else:
#            self.sign = 1
#
#        self.weight = weight
#
#    def __call__(self, channel, input, target):
#        d = channel.d
#        loss = effective_rank(channel)/d**2 - self.weight*tf.cast(attraction(channel, N=10000), dtype = precision)
#        loss = self.sign*loss
#
#        return loss


class AttractionRankTradeoff:
    """Optimize towards channel with high(low) effective rank,
    and low(high) attraction"""

    def __init__(self, weight=1):
        self.weight = weight

    def __call__(self, channel, input, target):
        a = target[0]
        loss = (effective_rank(channel) - a) ** 2 + self.weight * tf.cast(
            attraction(channel, N=1000), dtype=precision
        )

        return loss


class Conj2:
    """Optimize towards channel that breaks conj. 2"""

    def __init__(self, index):
        self.index = index

    def __call__(self, channel, input, target):
        d = channel.d
        spectrum = channel_spectrum(channel, real=True)
        x = spectrum[:, 0]
        loss = d * (d - 1) + d * x[self.index] - tf.math.reduce_sum(x)

        return loss


class Conj3:
    """Optimize towards channel that breaks conj. 2"""

    def __init__(self, index, sign=1):
        self.index = index
        self.sign = sign

    def __call__(self, channel, input, target):
        d = channel.d
        spectrum = channel_spectrum(channel, real=False)
        loss = self.sign * self._conjecture(spectrum)

        return loss[0]

    def _conjecture(self, spectrum):
        d = int(np.sqrt(spectrum.shape[0]))
        z = spectrum
        return tf.abs(z[self.index]) ** d - tf.abs(tf.math.reduce_prod(z))


##############################


class MininumEigenvalue:
    """Penalize large minimum eigenvalue"""

    def __call__(self, channel, input, target):
        state = input[0]
        state = channel.apply_channel(state)
        eig, _ = tf.linalg.eigh(state)
        loss = tf.math.reduce_min(eig)
        return loss[0]


class MinimizeEigenvalues:
    """Penalize sum of eigenvalues"""

    def __call__(self, channel, input, target):
        state = input[0]
        state = channel.apply_channel(state)
        eig, _ = tf.linalg.eigh(state)
        loss = tf.math.reduce_sum(eig)
        return loss[0]


class MininumAverageEigenvalue:
    """Penalize large average eigenvalue"""

    def __call__(self, channel, input, target):
        state = input[0]
        state = channel.apply_channel(state)
        eig, _ = tf.linalg.eigh(state)
        loss = tf.math.reduce_mean(eig)
        return loss[0]
