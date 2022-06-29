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
        self.loss = loss
        self.input_list = input_list
        self.target_list = target_list
        self.input_val_list = input_val_list
        self.target_val_list = target_val_list
        self.optimizer = optimizer


#   @profile
    def train(self, num_iter, N = 1):
        index_list = list(range(len(self.input_list)))

        for step in tqdm(range(num_iter)):
            random.shuffle(index_list)
            batch_list = index_list[:N]

            with tf.GradientTape() as tape:
                self.q_map.generate_map()
                loss = sum([self.loss(self.q_map, self.input_list[index], self.target_list[index]) for index in batch_list])/N

            grads = tape.gradient(loss, self.q_map.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.q_map.parameter_list))

            loss = sum([self.loss(self.q_map, self.input_list[index], self.target_list[index]) for index in index_list])/len(index_list)
            print(step, loss.numpy())
