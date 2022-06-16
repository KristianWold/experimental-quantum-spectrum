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
from cost_functions import *

def maps_to_choi(map_list):
    d = map_list[0].d
    choi = np.zeros((d**2, d**2), dtype="complex128")
    for i in range(d):
        for j in range(d):
            M = np.zeros((d,d), dtype="complex128")
            M[i,j] = 1
            M_prime = np.copy(M)
            for map in map_list:
                M_prime = map.apply_map(M_prime)

            choi += np.kron(M_prime, M)

    return choi


def channel_fidelity(map_A, map_B):
    choi_A = maps_to_choi([map_A])
    choi_B = maps_to_choi([map_B])
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def reshuffle_choi(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)
    return choi


def choi_spectrum(choi):
    choi = reshuffle_choi(choi)
    eig, _ = np.linalg.eig(choi)

    x = np.real(eig)
    y = np.imag(eig)

    return np.array([x, y])


class ChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(d**2, rank)
        self.parameter_list = [self.A, self.B]

        self.choi = None
        self.generate_map()
        self.k = np.array([[-10]], dtype = "float64")

    def generate_map(self):
        I = np.eye(self.d)
        X = self.A + 1j*self.B
        XX = X@X.conj().T
        #partial trace
        Y = partial_trace(XX)
        Y = sqrtm(Y)
        Y = np.linalg.inv(Y)
        Ykron = np.kron(I, Y)

        #choi
        self.choi = Ykron@XX@Ykron

    def apply_map(self, state):
        #reshuffle
        choi = reshuffle_choi(self.choi)

        #flatten
        state = state.reshape(-1, 1)

        state = (choi@state).reshape(self.d, self.d)
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter -= weight_gradient

        self.generate_map()


class DoubleChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        self.double_choi = [ChoiMap(d, rank), ChoiMap(d, rank)]
        self.parameter_list = []
        self.parameter_list.extend(self.double_choi[0].parameter_list)
        self.parameter_list.extend(self.double_choi[1].parameter_list)


        self.generate_map()
        self.k = np.array([[-10]], dtype = "float64")

    def generate_map(self):
        self.double_choi[0].generate_map()
        self.double_choi[1].generate_map()

    def apply_map(self, state):
        #reshuffle
        state = self.double_choi[0].apply_map(state)
        state = self.double_choi[1].apply_map(state)
        return state

    def update_parameters(self, weight_gradient_list):
        self.double_choi[0].update_parameters(weight_gradient_list[:2])
        self.double_choi[1].update_parameters(weight_gradient_list[2:])


class SquareRootChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        self.square_root_choi = ChoiMap(d, rank)
        self.parameter_list = self.square_root_choi.parameter_list


        self.generate_map()
        self.k = np.array([[-10]], dtype = "float64")

    def generate_map(self):
        self.square_root_choi.generate_map()

    def apply_map(self, state):
        #reshuffle
        state = self.square_root_choi.apply_map(self.square_root_choi.apply_map(state))
        return state

    def update_parameters(self, weight_gradient_list):
        self.square_root_choi.update_parameters(weight_gradient_list)


class KrausMap():

    def __init__(self,
                 U=None,
                 c=None,
                 d=None,
                 rank = None,
                 generate_map = True):
        self.U = U
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(rank*d, d)
        self.parameter_list = [self.A, self.B]

        if self.U is not None:
            k = -np.log(1/c - 1)
            self.k = np.array([[k]], dtype = "float64")
            self.parameter_list.append(self.k)

        self.kraus_list = None
        if generate_map:
            self.generate_map()

    def generate_map(self, U=None):
        d = self.d
        X = self.A + 1j*self.B
        if U is None:
            U = generate_unitary(X)

        self.kraus_list = []
        if self.U is not None:
            c = 1/(1 + np.exp(-self.k[0,0]))
            self.kraus_list.append(np.sqrt(c)*self.U)
        else:
            c = 0

        self.kraus_list.extend([np.sqrt(1-c)*U[i*d:(i+1)*d, :d] for i in range(self.rank)])

    def apply_map(self, state):

        state = sum([K@state@K.T.conj() for K in self.kraus_list])
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter -= weight_gradient

        self.generate_map()
