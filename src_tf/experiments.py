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
from utils import *
from set_precision import *

#@profile
def prepare_input(config, return_mode = "density"):
    """0 = |0>, 1 = |1>, 2 = |+>, 3 = |->, 4 = |i+>, 5 = |i+>"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config):
        if gate == 0:
            pass
        if gate == 1:
            circuit.rx(np.pi, i)
        if gate == 2:
            circuit.ry(np.pi/2, i)
        if gate == 3:
            circuit.ry(-np.pi/2, i)
        if gate == 4:
            circuit.rx(-np.pi/2, i)
        if gate == 5:
            circuit.rx(np.pi/2, i)

    if return_mode == "density":
        state = DensityMatrix(circuit.reverse_bits()).data
    if return_mode == "unitary":
        state = Operator(circuit.reverse_bits()).data
    if return_mode == "circuit":
        state = circuit.reverse_bits()

    if return_mode == "circuit_measure":
        circuit.add_register(qk.ClassicalRegister(n))
        circuit.measure(circuit.qregs[0], circuit.cregs[0])
        state = circuit.reverse_bits()


    return state


def pauli_observable(config, return_mode = "density"):

    n = len(config)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    basis = [X, Y, Z, I]

    if return_mode == "density":
        string = [basis[idx] for idx in config]
        result = kron(*string)

    q_reg = qk.QuantumRegister(n)
    c_reg = qk.ClassicalRegister(n)
    circuit = qk.QuantumCircuit(q_reg, c_reg)

    for i, index in enumerate(config):
        if index == 0:
            circuit.ry(-np.pi/2, i)

        if index == 1:
            circuit.rx(np.pi/2, i)

        if index == 2:
            pass    #measure in computational basis

    if return_mode == "circuit":
        circuit.measure(q_reg, c_reg)
        result = circuit.reverse_bits()

    if return_mode == "unitary":
        trace_index_list = []

        for i, idx in enumerate(config):
            if idx == 3:
                trace_index_list.append(i)

        observable = parity_observable(n, trace_index_list)

        result = [Operator(circuit.reverse_bits()).data, observable]

    return result


def generate_pauli_circuits(n = None,
                            circuit_target = None,
                            N = None,
                            trace=False):
    state_index, observ_index = index_generator(n, N, trace=trace)

    if trace:
        num_observ = 4
    else:
        num_observ = 3

    input_list = [[],[]]
    circuit_list = []
    for i, j in zip(state_index, observ_index):

        config1 = numberToBase(i, 6, n)
        U_prep = prepare_input(config1, return_mode = "unitary")

        config2 = numberToBase(j, num_observ, n)
        U_basis, _ = pauli_observable(config2, return_mode = "unitary")


        input_list[0].append(U_prep)
        input_list[1].append(U_basis)

        if circuit_target is not None:
            state_circuit = prepare_input(config1, return_mode = "circuit")
            observable_circuit = pauli_observable(config2, return_mode = "circuit")

            circuit = state_circuit
            circuit.barrier()
            circuit = circuit.compose(circuit_target)
            circuit.barrier()
            circuit.add_register(observable_circuit.cregs[0])
            circuit = circuit.compose(observable_circuit)

            circuit_list.append(circuit)

    input_list[0] = tf.convert_to_tensor(input_list[0], dtype=precision)
    input_list[1] = tf.convert_to_tensor(input_list[1], dtype=precision)

    return input_list, circuit_list


def generate_pauliInput_circuits(n = None):
    input_list = []
    circuit_list = []
    for i in range(6**n):

        config = numberToBase(i, 6, n)
        U_prep = prepare_input(config, return_mode = "unitary")
        circuit = prepare_input(config, return_mode = "circuit_measure")

        input_list.append(U_prep)
        circuit_list.append(circuit)

    input_list = tf.convert_to_tensor(input_list, dtype=precision)

    return input_list, circuit_list


def generate_bitstring_circuits(n):
    circuit_list = []
    for i in range(2**n):
        q_reg = qk.QuantumRegister(n)
        c_reg = qk.ClassicalRegister(n)
        circuit = qk.QuantumCircuit(q_reg, c_reg)
        config = numberToBase(i, 2, n)
        for j, index in enumerate(config):
            if index:
                circuit.x(j)
        circuit.measure(q_reg, c_reg)
        circuit_list.append(circuit.reverse_bits())

    return circuit_list


def generate_corruption_matrix(counts_list):
    n = len(list(counts_list[0].keys())[0])
    corr_mat = np.zeros((2**n, 2**n))
    for i, counts in enumerate(counts_list):
        for string, value in counts.items():
            index = int(string, 2)
            corr_mat[index, i] = value

    corr_mat = corr_mat/sum(counts_list[0].values())
    return corr_mat


def counts_to_probs(counts_list):
    N = len(counts_list)
    n = len(list(counts_list[0].keys())[0])
    probs = np.zeros((N, 2**n))
    for i in range(N):
        for string, value in counts_list[i].items():
            index = int(string, 2)
            probs[i, index] = value
    probs = probs/sum(counts_list[0].values())
    probs = tf.convert_to_tensor(probs, dtype=precision)
    return probs


def corr_mat_to_povm(corr_mat):
    d = corr_mat.shape[0]
    povm = []
    for i in range(d):
        M = tf.linalg.diag(corr_mat[i,:])
        povm.append(M)

    povm = tf.convert_to_tensor(povm, dtype=precision)

    return povm


def init_ideal(d):
    init = np.zeros((d,d))
    init[0, 0] = 1
    init = tf.convert_to_tensor(init, dtype = precision)
    return init


def povm_ideal(d):
    povm = corr_mat_to_povm(np.eye(d))
    return povm


def measurement(state, U_basis=None, povm=None):
    d = state.shape[1]
    if U_basis is None:
        U_basis = tf.eye(d, dtype=precision)
        U_basis = tf.expand_dims(U_basis, axis = 0)

    if povm is None:
        povm = corr_mat_to_povm(np.eye(d))

    Ustate = tf.matmul(U_basis, state)
    UstateU = tf.matmul(Ustate, U_basis, adjoint_b=True)

    state = tf.expand_dims(UstateU, axis=1)
    povm = tf.expand_dims(povm, axis=0)

    probs = tf.linalg.trace(state@povm)

    return probs


def parity_observable(n, trace_index_list=[]):
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    observable = n*[Z]
    for index in trace_index_list:
        observable[index] = I

    observable = np.diag(kron(*observable))
    return observable


def variational_circuit(n):
    theta = np.random.uniform(-np.pi, np.pi, 2*n)
    circuit = qk.QuantumCircuit(n)
    for i, angle in enumerate(theta[:n]):
        circuit.ry(angle, i)

    for i in range(n-1):
        circuit.cnot(i, i+1)

    for i, angle in enumerate(theta[n:2*n]):
        circuit.rx(angle, i)

    for i in reversed(range(n-1)):
        circuit.cnot(i+1, i)

    return circuit




class SPAM:
    def __init__(self,
                 d=None,
                 init = "random",
                 povm = "random",
                 use_corr_mat = False,
                 optimizer = None):

        self.d = d
        self.use_corr_mat = use_corr_mat

        self.parameter_list = []
        if init is "random":
            self.A = tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
            self.B = tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
            self.parameter_list.extend([self.A, self.B])
        elif init is "ideal":
            self.A = np.zeros((d,d))
            self.A[0,0] = 1
            self.A = tf.Variable(tf.cast(self.A, dtype = precision))
            self.B = tf.Variable(tf.zeros_like(self.A, dtype = precision))
            self.parameter_list.extend([self.A, self.B])
        else:
            self.A = self.B = None
            self.init = init

        if povm is "random":
            if not use_corr_mat:
                self.C = tf.Variable(tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision))
                self.D = tf.Variable(tf.cast(tf.random.normal((d, d, d), 0, 1), dtype = precision))
                self.parameter_list.extend([self.C, self.D])
            else:
                self.C =  tf.Variable(tf.cast(tf.random.normal((d, d), 0, 1), dtype = precision))
                self.parameter_list.extend([self.C])
        elif povm is "ideal":
            if not use_corr_mat:
                self.C = np.zeros((d,d,d))
                for i in range(d):
                    self.C[i,i,i] = 1
                self.C = tf.Variable(tf.cast(self.C, dtype = precision))
                self.D = tf.Variable(tf.zeros_like(self.C, dtype = precision))
                self.parameter_list.extend([self.C, self.D])
            else:
                self.C =  tf.Variable(tf.cast(tf.eye(d), dtype = precision))
                self.parameter_list.extend([self.C])

        else:
            self.C = self.D = None
            self.povm = povm

        self.optimizer = optimizer
        self.generate_SPAM()

    def generate_SPAM(self):
        if self.A is not None:
            X = self.A + 1j*self.B
            XX = tf.matmul(X, X, adjoint_b=True)
            state = XX/tf.linalg.trace(XX)
            self.init = state

        if self.C is not None:
            if not self.use_corr_mat:
                X = self.C + 1j*self.D
                XX = tf.matmul(X, X, adjoint_b=True)
                D = tf.math.reduce_sum(XX, axis = 0)
                invsqrtD = tf.linalg.inv(tf.linalg.sqrtm(D))
                self.povm = tf.matmul(tf.matmul(invsqrtD, XX), invsqrtD)

            else:
                X = tf.abs(self.C)
                X = X/tf.reduce_sum(X, axis = 1)
                corr_mat = tf.transpose(X)
                self.povm = corr_mat_to_povm(corr_mat)

    def train(self, num_iter, inputs, targets, N = None, verbose = True):
        if N is None:
            N = targets.shape[0]
        indices = tf.range(targets.shape[0])

        for step in tqdm(range(num_iter)):
            batch = tf.random.shuffle(indices)[:N]
            inputs_batch = tf.gather(inputs, batch, axis=0)
            targets_batch = tf.gather(targets, batch, axis=0)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.parameter_list)
                self.generate_SPAM()
                outputs = measurement(tf.repeat(self.init[None,:,:], N, axis=0),
                                      U_basis = inputs_batch,
                                      povm = self.povm)

                loss = self.d**2*tf.math.reduce_mean((outputs - targets_batch)**2)

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            if verbose:
                print(step, np.abs(loss.numpy()))

        self.generate_SPAM()

    def pretrain(self, num_iter, targets, verbose = True):
        init_target, povm_target = targets
        for step in tqdm(range(num_iter)):

            with tf.GradientTape() as tape:
                self.generate_SPAM()
                loss1 = tf.reduce_mean(tf.abs(self.init - init_target)**2)
                loss2 = tf.reduce_mean(tf.abs(self.povm - povm_target)**2)
                loss = loss1 + loss2

            grads = tape.gradient(loss, self.parameter_list)
            self.optimizer.apply_gradients(zip(grads, self.parameter_list))
            if verbose:
                print(step, np.abs(loss.numpy()))

        self.generate_SPAM()
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))
