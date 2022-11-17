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
    def __init__(self, sample_freq=100, loss_function=None, verbose=True):
        self.sample_freq = sample_freq
        self.loss_function = loss_function
        self.verbose = verbose

        self.loss_train_list = []
        self.loss_val_list = []

    def log(self, other, push=False):
        if other.counter % self.sample_freq == 0 or push:
            loss_train = None
            # loss_train = np.real(
            #    self.loss_function(other.channel, other.inputs, other.targets).numpy()
            # )
            self.loss_train_list.append(loss_train)

            loss_val = None
            if other.targets_val != None:
                loss_val = np.real(
                    self.loss_function(
                        other.channel, other.inputs_val, other.targets_val
                    ).numpy()
                )

            self.loss_val_list.append(loss_val)
            if self.verbose or push:
                print(loss_train, loss_val)


class ModelQuantumMap:
    def __init__(
        self,
        channel=None,
        loss_function=None,
        optimizer=None,
        logger=None,
    ):
        self.channel = channel
        self.loss_function = loss_function
        self.optimizer = optimizer

        if logger is None:
            logger = Logger(loss_function=loss_function, verbose=False)
        self.logger = logger

        if not isinstance(self.loss_function, list):
            self.loss_function = [self.loss_function]

    #   @profile
    #   @tf.function
    def train(
        self,
        inputs=None,
        targets=None,
        inputs_val=None,
        targets_val=None,
        num_iter=1000,
        N=0,
    ):

        self.inputs = inputs
        self.targets = targets
        self.inputs_val = inputs_val
        self.targets_val = targets_val
        self.counter = 0

        if N != 0:
            indices = list(range(targets.shape[0]))

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

            self.logger.log(self)
            self.counter += 1

        self.channel.generate_channel()
        self.logger.log(self, push=True)

    def zero_optimizer(self):
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))

    def set_loss_function(self, loss_function, zero_optimizer=True):
        self.loss_function = loss_function
        if not isinstance(self.loss_function, list):
            self.loss_function = [self.loss_function]
        if zero_optimizer:
            self.zero_optimizer()
