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
from experiments import *
from set_precision import *
from quantum_maps import *


def state_fidelity_loss(q_map, input, target, grad=False):
    state = input
    output = q_map.apply_map(input)
    loss = -state_fidelity(output, target)
    return loss


def expectation_value_loss(q_map, input, target, grad=False):
    state, observable = input
    state = q_map.apply_map(state)
    output = expectation_value(state, observable)
    loss = tf.abs(output - target)**2
    return loss


class ProbabilityLoss:
    def __init__(self, reg = 0):
        self.reg = reg

    def __call__(self, q_map, input, target):
        N = target.shape[0]
        d = q_map.spam.init.shape[0]
        U_prep, U_basis = input

        state = tf.repeat(tf.expand_dims(q_map.spam.init, axis=0), N, axis=0)
        state = apply_unitary(state, U_prep)
        state = q_map.apply_map(state)
        output = measurement(state, U_basis, q_map.spam.povm)
        loss = d**2*tf.math.reduce_mean((output - target)**2)
        if self.reg != 0:
            rank_eff = effective_rank(q_map)
            loss += self.reg*rank_eff
        return loss


class RankShrink:

    def __init__(self, rank=None, inflate=False):
        if inflate:
            self.sign = -1
        else:
            self.sign = 1

    def __call__(self, q_map, input, target):   
        loss = self.sign*effective_rank(q_map)

        return loss


class RankMSE:
    def __call__(self, q_map, input, target):
        
        rank_target = target
        loss = (effective_rank(q_map) - rank_target)**2

        return loss


class Conj2:
    def __init__(self, index):
        self.index = index

    def __call__(self, q_map, input, target):
        d = q_map.d
        choi = kraus_to_choi(q_map)
        spectrum = choi_spectrum(choi, real = True)
        x = spectrum[:,0]
        loss = (d*(d-1) + d*x[self.index] - tf.math.reduce_sum(x))

        return loss


class RankMSE:
    def __call__(self, q_map, input, target):
        
        rank_target = target
        loss = (effective_rank(q_map) - rank_target)**2

        return loss


def channel_fidelity_loss(q_map, input, target, grad=False):
    q_map_target = target
    loss = -channel_fidelity(q_map, q_map_target)
    return loss


def channel_norm_loss(q_map, input, target, grad=False):
    q_map_target = target
    choi_model = kraus_to_choi(q_map)
    choi_target = kraus_to_choi(q_map_target)

    loss = tf.math.reduce_sum(tf.abs(choi_model - choi_target)**2)
    return loss


class SpectrumDistance():

    def __init__(self, sigma = 0.1, k = 1000):
        self.sigma = sigma
        self.sigma_ = sigma
        self.k = k
        self.mode = "density"
        self.t = 0

    def __call__(self, q_map, input, target, grad=False):
        spectrum_target = input[0]

        choi_model = kraus_to_choi(q_map)
        spectrum_model = choi_spectrum(choi_model, real=True)

        if self.mode == "density":
            loss = self.overlap(spectrum_model, spectrum_model)
            loss += -2*self.overlap(spectrum_model, spectrum_target)
            loss += self.overlap(spectrum_target, spectrum_target)

        if self.mode == "pairwise":
            connections = greedy_pair_distance(spectrum_model, spectrum_target)
            loss = pair_distance(spectrum_model, spectrum_target, connections)
        
        self.t += 1
        self.sigma = np.sqrt(self.k)*self.sigma_/np.sqrt(self.k + self.t)

        return loss

    def overlap(self, spectrum_a, spectrum_b):
        aa = tf.math.reduce_sum(spectrum_a*spectrum_a, axis=1, keepdims=True)
        bb = tf.math.reduce_sum(spectrum_b*spectrum_b, axis=1, keepdims=True)
        ab = tf.matmul(spectrum_a, spectrum_b, adjoint_b=True)

        expo = aa - 2*ab + tf.transpose(bb)
        sum = 1/np.sqrt(self.sigma)*tf.math.reduce_mean(tf.math.exp(-expo/self.sigma**2))
        
        return sum    
 
    def greedy_pair_distance(self, spectrum_a, spectrum_b):
        connections = []
        not_connected = len(spectrum_a)*[True]

        for i, a in enumerate(spectrum_a):
            min_dist = float("inf")
            idx = 0
            for j, b in enumerate(spectrum_b):
                dist = tf.abs((a[0] - b[0])**2 + (a[1] - b[1])**2)
                if (dist < min_dist) and not_connected[j]:

                    min_dist = dist
                    idx = j

            not_connected[idx] = False
            connections.append(idx)

        return connections

    def pair_distance(self, spectrum_a, spectrum_b, connections):
        distance = 0
        for i, idx in enumerate(connections):
            distance += (spectrum_a[i][0] - spectrum_b[idx][0])**2 + (spectrum_a[i][1] - spectrum_b[idx][1])**2

        return distance


def min_pair_distance(a_list = None,
                      b_list = None,
                      connections = None,
                      num_iter = 100,
                      T = 1):

    length = len(a_list)
    if connections is None:
        connections = greedy_pair_distance(a_list, b_list)

    distance = pair_distance(a_list, b_list, connections)
    distance_list = [distance]

    for i in range(num_iter):
        idx1 = random.randint(0, length-1)
        idx2 = random.randint(0, length-2)
        if idx1 <= idx2:
            idx2 += 1

        distance_old = np.linalg.norm(a_list[idx1] - b_list[connections[idx1]]) \
                     + np.linalg.norm(a_list[idx2] - b_list[connections[idx2]])
        distance_new = np.linalg.norm(a_list[idx2] - b_list[connections[idx1]]) \
                     + np.linalg.norm(a_list[idx1] - b_list[connections[idx2]])
        distance_diff = distance_new - distance_old


        u = random.uniform(0,1)
        if np.exp(-distance_diff/T) > u:
            distance += distance_diff
            connections[idx1], connections[idx2] = connections[idx2], connections[idx1]

        distance_list.append(distance)

    return connections, distance_list
