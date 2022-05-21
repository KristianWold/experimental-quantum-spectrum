import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from scipy.linalg import cholesky
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
    X = X.reshape(d,d,d,d)
    if discard_first:
        Y = np.einsum("ijik->jk", X)
    else:
        Y = np.einsum("jiki->jk", X)
    return Y


#@profile
def state_fidelity(A, B):

    sqrtB = sqrtm(B)
    C = sqrtB@A@sqrtB

    fidelity = np.trace(sqrtm(C))
    return np.abs(fidelity)

def state_norm(A, B):

    norm = 1 - np.linalg.norm(A - B)
    return np.abs(norm)


#@profile
def prepare_input(config, return_mode = "density"):
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

    if return_mode == "density":
        state = DensityMatrix(circuit).data
    if return_mode == "unitary":
        state = Operator(circuit).data
    if return_mode == "circuit":
        state = circuit

    return state


def sample_bitstring(state, basis, return_index):
    N = state.shape[0]
    n = np.log(N)
    state = basis@state@basis.T.conj()
    probs = np.abs(np.diag(state))
    index = np.random.choice(range(N), p = probs)

    if return_index:
        output = index
    else:
        output = numberToBase(index, 2, n)

    return output


def likelihood(state, data):
    pass


#@profile
def generate_ginibre(dim1, dim2):
    A = np.random.normal(0, 1, (dim1, dim2))
    B = np.random.normal(0, 1, (dim1, dim2))
    X = A + 1j*B
    return X, A, B

def generate_state(dim1, dim2):
    X, _, _ = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/np.trace(X@X.conj().T)
    return state


def generate_unitary(X):
    Q, R = np.linalg.qr(X)
    R = np.diag(R)
    sign = R/np.abs(R)
    U = Q@np.diag(sign)

    return U


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
        d = self.d

        #reshuffle
        choi = self.choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)

        #flatten
        state = state.reshape(-1, 1)

        state = (choi@state).reshape(d, d)
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter += weight_gradient

        self.generate_map()


class KrausMap():

    def __init__(self, U, c, d, rank):
        self.U = U
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(rank*d, d)
        k = -np.log(1/c - 1)
        self.k = np.array([[k]], dtype = "float64")
        self.parameter_list = [self.A, self.B, self.k]

        self.kraus_list = None
        self.generate_map()

    def generate_map(self):
        d = self.d
        X = self.A + 1j*self.B
        U = generate_unitary(X)
        self.kraus_list = [U[i*d:(i+1)*d, :d] for i in range(self.rank)]

    def apply_map(self, state):
        c = 1/(1 + np.exp(-self.k[0,0]))
        state = c*self.U@state@self.U.T.conj() + (1 - c)*sum([K@state@K.T.conj() for K in self.kraus_list])
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter += weight_gradient

        self.generate_map()


class ModelQuantumMap:

    def __init__(self, model, state_input_list, state_target_list, lr, h):
        self.model = model
        self.state_input_list = state_input_list
        self.state_target_list = state_target_list
        self.lr = lr
        self.h = h

        self.d = model.d
        self.rank = model.rank

        self.adam = Adam()
        self.fid_list = []

#    @profile
    def train(self, num_iter, use_adam=False):

        for step in tqdm(range(num_iter)):
            index = np.random.randint(len(self.state_input_list))
            self.state_input = self.state_input_list[index]
            self.state_target = self.state_target_list[index]

            state_model = self.model.apply_map(self.state_input)
            self.fid_zero = state_fidelity(state_model, self.state_target)

            grad_list = []
            for parameter in self.model.parameter_list:
                grad_matrix = self.calculate_gradient(parameter)
                grad_list.append(grad_matrix)

            if use_adam:
                grad_list = self.adam(grad_list, self.lr)

            self.model.update_parameters(grad_list)
            c = 1/(1 + np.exp(-self.model.k[0,0]))

            self.fid_list.append(self.fid_zero)
            print(f"{step}: fid: {self.fid_zero:.3f}, c: {c:.3f}")

#    @profile
    def calculate_gradient(self, parameter):

        h = self.h
        state_input = self.state_input
        state_target = self.state_target

        grad_matrix = np.zeros_like(parameter)
        for i in range(parameter.shape[0]):
            for j in range(parameter.shape[1]):
                parameter[i, j] += h
                self.model.generate_map()
                state_plus = self.model.apply_map(state_input)
                fid_plus = state_fidelity(state_plus, state_target)
                parameter[i, j] -= h

                grad_matrix[i, j] = (fid_plus-self.fid_zero)/h

        return grad_matrix
