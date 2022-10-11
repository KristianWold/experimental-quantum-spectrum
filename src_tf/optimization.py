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
        self.loss_train = []
        self.loss_val = []
        self.c_list = []


#   @profile
    def train(self,
              inputs = None,
              targets = None,
              inputs_val = None,
              targets_val = None,
              num_iter = 1000,
              N = None,
              verbose = True,
              use_batch = True):

        if use_batch:
            indices = tf.range(targets.shape[0])
            if N is None:
                N = targets.shape[0]

        for step in tqdm(range(num_iter)):
            if use_batch:
                batch = tf.random.shuffle(indices)[:N]
                inputs_batch = [tf.gather(data, batch, axis=0) for data in inputs]
                targets_batch = tf.gather(targets, batch, axis=0)
            else:
                inputs_batch = inputs
                targets_batch = targets

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.q_map.parameter_list)
                self.q_map.generate_map()
                loss = self.loss(self.q_map, inputs_batch, targets_batch)

            grads = tape.gradient(loss, self.q_map.parameter_list)
            #for i in range(len(grads)):
            #    grads[i] = grads[i] + tf.cast(tf.random.normal(grads[i].shape, 0, 0.01), dtype = precision) 
        
            self.optimizer.apply_gradients(zip(grads, self.q_map.parameter_list))

            if targets_val is None:
                loss_val = 0
            elif len(targets_val) == 1:
                loss_val = channel_fidelity(self.q_map, targets_val[0])
            else:
                loss_val = np.abs(self.loss(self.q_map, inputs_val, targets_val).numpy())

            self.loss_train.append(np.abs(loss.numpy()))
            self.loss_val.append(loss_val)

            if self.q_map.c is not None:
                self.c_list.append(np.abs(self.q_map.c.numpy()))
            
            if verbose:
                print(f"Step:{step}, train: {np.abs(loss.numpy()):.5f}, val: {loss_val:.5f}")
            
        print(np.abs(loss.numpy()), loss_val)
        self.q_map.generate_map()
    
    def zero_optimizer(self):
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))



class POVM:
    def __init__(self, d, optimizer, trainable=True):
        self.d = d
        self.A = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)
        self.B = tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision)

        if trainable:
            self.A = tf.Variable(self.A, trainable = True)
            self.B = tf.Variable(self.B, trainable = True)

        self.parameter_list = [self.A, self.B]
        self.optimizer = optimizer

        self.generate_POVM()

    def generate_POVM(self):
        G = self.A + 1j*self.B
        AA = tf.matmul(G, G, adjoint_b=True)
        D = tf.math.reduce_sum(AA, axis = 0)
        invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
        self.povm = tf.matmul(tf.matmul(invsqrtD, AA), invsqrtD)

    def train(self, num_iter, inputs, targets, N = 1):
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape() as tape:
                self.generate_POVM()
                outputs = measurement(inputs_batch, povm=self.povm)

                loss = self.d**2*tf.math.reduce_mean((outputs - targets_batch)**2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            print(step, np.abs(loss.numpy()))

        self.generate_POVM()
