import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm


def numberToBase(n, b, num_digits):
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits)<num_digits:
        digits.append(0)
    return digits[::-1]

#@profile
def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = tf.reshape(X, (d,d,d,d))
    if discard_first:
        Y = tf.einsum("ijik->jk", X)
    else:
        Y = tf.einsum("jiki->jk", X)
    return Y

#@profile
def state_fidelity(A, B):
    sqrtB = tf.linalg.sqrtm(B)
    fidelity = tf.linalg.trace(tf.linalg.sqrtm(sqrtB@A@sqrtB))
    return fidelity

#@profile
def apply_map(state, choi):
    d = state.shape[0]

    #reshuffle
    choi = tf.reshape(choi, (d,d,d,d))
    choi = tf.transpose(choi, perm = [0,2,1,3])
    choi = tf.reshape(choi, (d**2,d**2))

    #flatten
    state = tf.reshape(state, (d**2, 1))

    state = tf.reshape(choi@state, (d, d))
    return state

#@profile
def prepare_input(config, return_unitary = False):
    """1 = |0>, 2 = |1>, 3 = |+>, 4 = |->, 5 = |+i>, 6 = |-i>"""
    n = len(config)
    circuit = qk.QuantumCircuit(n)
    for i, gate in enumerate(config):
        if gate == 0:
            circuit.i(i)
        if gate == 1:
            circuit.x(i)
        if gate == 2:
            circuit.h(i)
        if gate == 3:
            circuit.x(i)
            circuit.h(i)
        if gate == 4:
            circuit.h(i)
            circuit.s(i)
        if gate == 5:
            circuit.x(i)
            circuit.h(i)
            circuit.s(i)

    if not return_unitary:
        state = tf.convert_to_tensor(DensityMatrix(circuit).data, dtype=tf.complex128)
    else:
        state = Operator(circuit).data

    return state


def bitstring_density(state, basis):

    return np.abs(np.diag(basis.conj().T@state@basis))

#@profile
def generate_ginibre(dim1, dim2, real = False):
    ginibre = np.random.normal(0, 1, (dim1, dim2))
    if not real:
         ginibre = ginibre + 1j*np.random.normal(0, 1, (dim1, dim2))
    return tf.convert_to_tensor(ginibre, dtype=tf.complex128)


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/torch.trace(X@X.conj().T)
    return state

#@profile
def generate_choi(X):
    d = int(np.sqrt(X.shape[0]))  # dim of Hilbert space
    I = tf.eye(d, dtype = tf.complex128)
    XX = X@tf.linalg.adjoint(X)

    #partial trace
    Y = tf.linalg.sqrtm(tf.linalg.inv(partial_trace(XX)))
    Ykron = tf.experimental.numpy.kron(I, Y)

    #choi
    choi = Ykron@XX@Ykron

    return choi

class Adam():
    def __init__(self, dims, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0
        self.m = np.zeros(dims, dtype="complex128")
        self.v = np.zeros(dims, dtype="complex128")


    def __call__(self, gradient):
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.abs(gradient)**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        gradient_modified = m_hat / (np.sqrt(v_hat) + self.eps)

        return gradient_modified



class ModelQuantumMap:

    def __init__(self, n, rank, state_input_list, state_target_list, lr, h):
        self.n = n
        self.rank = rank
        self.state_input_list = state_input_list
        self.state_target_list = state_target_list
        self.lr = lr
        self.h = h

        self.d = 2**n
        self.X_model = generate_ginibre(self.d**2, self.rank)

        self.adam = Adam(dims = (self.d**2, self.rank))
        self.fid_list = []

    #@profile
    def train(self, num_iter, use_adam=False):

        for step in tqdm(range(num_iter)):
            index = np.random.randint(0, len(self.state_input_list)-1)
            self.state_input = self.state_input_list[index]
            self.state_target = self.state_target_list[index]

            # Finite difference over all parameters
            self.X_plus = self.X_model + self.h
            self.X_minus = self.X_model - self.h
            self.X_iplus = self.X_model + 1j*self.h
            self.X_iminus = self.X_model - 1j*self.h

            grad_matrix = np.zeros_like(self.X_model)
            for idx in tqdm(range(self.d**2*self.rank)):
                i = idx//self.rank
                j = idx%self.rank
                grad_matrix[i, j] = self.calculate_gradient(i, j)

            grad_matrix /= self.h
            if use_adam:
                grad_matrix = self.adam(grad_matrix)
            grad_matrix = tf.convert_to_tensor(grad_matrix)

            self.X_model = self.X_model + self.lr*grad_matrix

            choi_model = generate_choi(self.X_model)
            state_model = apply_map(self.state_input, choi_model)
            fid = state_fidelity(state_model, self.state_target)

            self.fid_list.append(fid)
            print(f"{step}: {fid:.3f}")

    #@profile
    def calculate_gradient(self, i, j):

        h = self.h
        state_input = self.state_input
        state_target = self.state_target

        mask = np.zeros_like(self.X_model, dtype="bool")
        mask[i, j] = True
        mask = tf.convert_to_tensor(mask)

        #Finite difference, real value
        X = tf.where(mask, self.X_plus, self.X_model)
        choi_plus = generate_choi(X)
        state_plus = apply_map(state_input, choi_plus)
        fid_plus = state_fidelity(state_plus, state_target)

        X = tf.where(mask, self.X_minus, self.X_model)
        choi_minus = generate_choi(X)
        state_minus = apply_map(state_input, choi_minus)
        fid_minus = state_fidelity(state_minus, state_target)

        grad = (fid_plus-fid_minus)

        #Finite difference, imaginary value
        X = tf.where(mask, self.X_iplus, self.X_model)
        choi_plus = generate_choi(X)
        state_plus = apply_map(state_input, choi_plus)
        fid_plus = state_fidelity(state_plus, state_target)

        X = tf.where(mask, self.X_iminus, self.X_model)
        choi_minus = generate_choi(X)
        state_minus = apply_map(state_input, choi_minus)
        fid_minus = state_fidelity(state_minus, state_target)

        grad += 1j*(fid_plus-fid_minus)
        return grad
