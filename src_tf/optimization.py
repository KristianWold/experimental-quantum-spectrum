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
from quantum_channel import *
from utils import *
from set_precision import *


class Logger:
    def __init__(self
                 ):
        
        self.loss_train = []
        self.loss_val = []

    def log(self, other):
        pass


class ModelQuantumMap:

    def __init__(self,
                 channel = None,
                 loss_function = None,
                 optimizer = None,
                 logger = None,
                 ):
        self.channel = channel
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.logger = logger

        if not isinstance(self.loss_function, list):
            self.loss_function = [self.loss_function]

#   @profile
#   @tf.function
    def train(self,
              inputs = None,
              targets = None,
              inputs_val = None,
              targets_val = None,
              num_iter = 1000,
              verbose = True,
              N = 0):

        self.inputs = inputs
        self.targets = targets
        self.inputs_val = inputs_val
        self.targets_val = targets_val

        if N != 0:
            indices = list(range(inputs.shape[0]))

        for step in tqdm(range(num_iter)):
            if N != 0:
                batch = tf.random.shuffle(indices)[:N]
                inputs_batch = [tf.gather(data, batch, axis=0) for data in inputs]
                targets_batch = tf.gather(targets, batch, axis=0)
            else:
                inputs_batch = inputs
                targets_batch = targets

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.channel.parameter_list)
                self.channel.generate_channel()
                
                loss = 0
                for loss_function in self.loss_function: 
                    loss += loss_function(self.channel, inputs_batch, targets_batch)

            grads = tape.gradient(loss, self.channel.parameter_list)
        
            self.optimizer.apply_gradients(zip(grads, self.channel.parameter_list))
            #self.logger.log(self)
            #if targets_val is None:
            #    loss_val = 0
            #elif len(targets_val) == 1:
            #    loss_val = channel_fidelity(self.channel, targets_val[0])
            #else:
            #   loss_val = np.abs(self.loss(self.channel, inputs_val, targets_val).numpy())

            #self.loss_train.append(loss.numpy())
            #self.loss_val.append(loss_val)

            #if self.channel.c is not None:
            #    self.c_list.append(np.abs(self.channel.c.numpy()))
            
            if verbose:
                print(f"Step:{step}, train: {np.real(loss.numpy())}")
            
        print(np.real(loss.numpy()))
        self.channel.generate_channel()
    
    def zero_optimizer(self):
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    def set_loss_function(self, loss_function, zero_optimizer=True):
        self.loss_function = loss_function
        if zero_optimizer:
            self.zero_optimizer()

