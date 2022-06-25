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
from copy import deepcopy


from quantum_tools import *
from loss_functions import *
from quantum_maps import *
from utils import *


class ModelQuantumMap:

    def __init__(self,
                 q_map,
                 loss,
                 input_list,
                 target_list,
                 input_val_list,
                 target_val_list,
                 optimizer,
                 ):
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


#   @profile
    def train(self, num_iter, use_adam=False, verbose=True, N = 1, choi_target=None):

        for step in tqdm(range(num_iter)):
            index_list = list(range(len(self.input_list)))
            random.shuffle(index_list)
            batch_list = index_list[:N]

            with tf.GradientTape() as tape:
                loss = self.loss(self.input[batch_list], self.target_list[batch_list])

            grads = tape.gradient(loss, self.q_map.parameters)
            self.optimizer.apply_gradients(zip(grads, self.q_map.parameters))
