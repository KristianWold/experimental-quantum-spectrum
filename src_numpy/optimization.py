import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from copy import deepcopy


from quantum_tools import *
from cost_functions import *
from quantum_maps import *
from utils import *


class Adam():

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = []
        self.v = []
        self.t = 0


    def __call__(self, weight_gradient_list, lr):
        if len(self.m) == 0:
            for weight_gradient in weight_gradient_list:
                self.m.append(np.zeros_like(weight_gradient))
                self.v.append(np.zeros_like(weight_gradient))

        self.t += 1
        weight_gradient_modified = []

        for grad, m_, v_ in zip(weight_gradient_list, self.m, self.v):
            m_[:] = self.beta1 * m_ + (1 - self.beta1) * grad
            v_[:] = self.beta2 * v_ + (1 - self.beta2) * grad*grad

            m_hat = m_ / (1 - self.beta1**self.t)
            v_hat = v_ / (1 - self.beta2**self.t)
            grad_modified = m_hat / (np.sqrt(v_hat) + self.eps)
            weight_gradient_modified.append(lr*grad_modified)

        return weight_gradient_modified

class ModelQuantumMap:

    def __init__(self,
                 q_map,
                 cost,
                 input_list,
                 target_list,
                 input_val_list,
                 target_val_list,
                 lr,
                 h):
        self.q_map = q_map
        self.cost = cost
        self.input_list = input_list
        self.target_list = target_list
        self.input_val_list = input_val_list
        self.target_val_list = target_val_list
        self.lr = lr
        self.h = h

        self.d = q_map.d
        self.rank = q_map.rank

        self.adam = Adam()
        self.fid_list = []
        self.c_list = []
        self.map_best = None
        self.cost_best = 1e10

#    @profile
    def train(self, num_iter, use_adam=False, verbose=True, N = 1, choi_target=None):

        self.cost_average = self.calculate_val_cost()

        for step in tqdm(range(num_iter)):

            grad_list = [np.zeros_like(parameter) for parameter in self.q_map.parameter_list]
            index_list = list(range(len(self.input_list)))
            random.shuffle(index_list)
            batch_index = index_list[:N]
            for index in batch_index:
                self.input = self.input_list[index]
                self.target = self.target_list[index]

                for parameter, grad in zip(self.q_map.parameter_list, grad_list):
                    grad_matrix = self.calculate_gradient(parameter)
                    grad += grad_matrix

            for grad in grad_list:
                grad /= N

            if use_adam:
                grad_list = self.adam(grad_list, self.lr)

            self.q_map.update_parameters(grad_list)

            self.cost_average = self.calculate_val_cost()
            if self.cost_average < self.cost_best:
                self.cost_best = self.cost_average
                self.map_best = deepcopy(self.q_map)


            c = 1/(1 + np.exp(-self.q_map.k[0,0]))

            if choi_target is not None:
                choi_model = kraus_to_choi([self.q_map])
                fid = state_fidelity(choi_model, choi_target)
            else:
                fid = self.cost_average

            self.fid_list.append(fid)
            self.c_list.append(c)
            if verbose:
                print(f"{step}: fid: {fid:.5f}, c: {c:.3f}")

    def calculate_val_cost(self):
        cost = sum([self.cost(self.q_map, input, target) for input, target in zip(self.input_val_list, self.target_val_list)])
        cost = cost/len(self.input_val_list)

        return cost

#    @profile
    def calculate_gradient(self, parameter):

        h = self.h
        input = self.input
        target = self.target

        grad_matrix = np.zeros_like(parameter)
        for i in range(parameter.shape[0]):
            for j in range(parameter.shape[1]):
                parameter[i, j] += h
                self.q_map.generate_map()
                cost_plus = self.cost(self.q_map, input, target, grad=True)

                parameter[i, j] -= 2*h
                self.q_map.generate_map()
                cost_minus = self.cost(self.q_map, input, target, grad=True)

                parameter[i, j] += h
                grad_matrix[i, j] = (cost_plus-cost_minus)/(2*h)

        return grad_matrix
