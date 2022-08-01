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
from set_precision import *


from quantum_tools import *
from loss_functions import *
from quantum_maps import *
from utils import *
from experiments import *
from set_precision import *


class ModelQuantumMap:

    def __init__(self,
                 q_map = None,
                 loss = None,
                 optimizer = None,
                 ):
        self.q_map = q_map
        self.loss = loss
        self.optimizer = optimizer
        self.loss_list = []
        self.c_list = []


#   @profile
    def train(self,
              inputs = None,
              targets = None,
              inputs_val = None,
              targets_val = None,
              num_iter = 1000,
              N = 1,
              verbose = True):

        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = [tf.gather(data, batch, axis=0) for data in inputs]
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.q_map.generate_map()
                loss = self.loss(self.q_map, inputs_batch, targets_batch)

            grads = tape.gradient(loss, self.q_map.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.q_map.parameter_list))

            if targets_val is None:
                pass
            elif len(targets_val) == 1:
                loss = channel_fidelity(self.q_map, targets_val[0])


            self.loss_list.append(np.abs(loss.numpy()))
            if self.q_map.c is not None: 
                self.c_list.append(np.abs(self.q_map.c.numpy()))

            if verbose:
                print(step, np.abs(loss.numpy()))
        
        self.q_map.generate_map()
