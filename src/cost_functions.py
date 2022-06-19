import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from quantum_tools import *


def state_density_loss(q_map, input, target, grad=False):
    state = input
    output = q_map.apply_map(input)
    cost = -state_fidelity(output, target)
    return cost


def expectation_value_loss(q_map, input, target, grad=False):
    state, observable = input
    state = q_map.apply_map(state)
    output = expectation_value(state, observable)
    cost = np.abs(output - target)**2
    return cost


def channel_fidelity_loss(q_map, input, target, grad=False):
    q_map_target = input
    cost = -channel_fidelity(q_map, q_map_target)
    return cost


class SpectrumDistance():

    def __init__(self, num_iter = 1000, T = 0):
        self.num_iter = num_iter
        self.T = T
        self.connections = None

    def __call__(self, q_map, input, target, grad=False):
        choi_model = maps_to_choi([q_map])

        q_map_target = input
        choi_target = maps_to_choi([q_map_target])

        spectrum_model = [np.array((a,b)) for a,b in zip(*choi_spectrum(choi_model))]
        spectrum_target = [np.array((a,b)) for a,b in zip(*choi_spectrum(choi_target))]

        if grad:
            num_iter = 0
        else:
            self.connections = greedy_pair_distance(spectrum_model, spectrum_target)
            num_iter = self.num_iter

        self.connections, distance_list = min_pair_distance(spectrum_model,
                                                            spectrum_target,
                                                            self.connections,
                                                            num_iter = num_iter,
                                                            T = self.T)

        cost = distance_list[-1]

        return cost


def greedy_pair_distance(a_list, b_list):
    connections = []
    not_connected = len(a_list)*[True]

    for i, a in enumerate(a_list):
        min_dist = float("inf")
        idx = 0
        for j, b in enumerate(b_list):
            dist = np.linalg.norm(a - b)
            if (dist < min_dist) and not_connected[j]:

                min_dist = dist
                idx = j

        not_connected[idx] = False
        connections.append(idx)

    return connections


def pair_distance(a_list,
                  b_list,
                  connections):
    distance = 0
    for i, idx in enumerate(connections):
        distance += np.linalg.norm(a_list[i] - b_list[idx])

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
