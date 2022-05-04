import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

np.set_printoptions(precision=2)

def numberToBase(n, b, num_digits):
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits)<num_digits:
        digits.append(0)
    return digits[::-1]


def partial_trace(X, discard_first = True):
    d = X.shape[0]
    d_red = int(np.sqrt(d))
    Y = np.zeros((d_red, d_red), dtype = "complex128")
    I = np.eye(d_red)

    for i in range(d_red):
        basis_vec = np.zeros((d_red, 1),  dtype = "complex128")
        basis_vec[i, 0] = 1

        if discard_first:
            basis_vec = np.kron(basis_vec, I)
        else:
            basis_vec = np.kron(I, basis_vec)

        Y = Y + basis_vec.T@X@basis_vec

    return Y


def state_fidelity(A, B):
    sqrtA = sqrtm(A)
    fidelity = np.trace(sqrtm(sqrtA@B@sqrtA))
    return np.abs(fidelity)


def apply_map(state, choi):
    d = state.shape[0]

    #reshuffle
    choi = choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)

    #flatten
    state = state.reshape(-1, 1)

    state = (choi@state).reshape(d, d)
    return state


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
        state = DensityMatrix(circuit).data
    else:
        state = Operator(circuit).data

    return state


def bitstring_density(state, basis):

    return np.abs(np.diag(basis.conj().T@state@basis))


def generate_ginibre(dim1, dim2, real = False):
    ginibre = np.random.normal(0, 1, (dim1, dim2))
    if not real:
         ginibre = ginibre + 1j*np.random.normal(0, 1, (dim1, dim2))
    return ginibre


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/np.trace(X@X.conj().T)
    return state


def generate_choi(X):
    d = int(np.sqrt(X.shape[0]))  # dim of Hilbert space
    I = np.eye(d)

    #partial trace
    Y = partial_trace(X@(X.conj().T), discard_first = True)
    sqrtYinv = np.linalg.inv(sqrtm(Y))

    #choi
    choi = np.kron(I, sqrtYinv)@X@(X.conj().T)@np.kron(I, sqrtYinv)

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


def calculate_gradient(inputs):
    self = inputs[0]
    j = inputs[1]

    h = self.h
    state_input = self.state_input
    state_target = self.state_target
    X_model = np.copy(self.X_model)
    gradient_vector = np.zeros(self.d**2, dtype = "complex128")
    for i in range(self.d**2):
        #Finite difference, real value
        X_model[i, j] += h
        choi_plus = generate_choi(X_model)
        state_plus = apply_map(state_input, choi_plus)
        fid_plus = state_fidelity(state_plus, state_target)

        X_model[i, j] -= 2*h
        choi_minus = generate_choi(X_model)
        state_minus = apply_map(state_input, choi_minus)
        fid_minus = state_fidelity(state_minus, state_target)
        X_model[i, j] += h

        grad = (fid_plus-fid_minus)/h
        gradient_vector[j] += grad

        #Finite difference, imaginary value
        X_model[i, j] += 1j*h
        choi_plus = generate_choi(X_model)
        state_plus = apply_map(state_input, choi_plus)
        fid_plus = state_fidelity(state_plus, state_target)

        X_model[i, j] -= 2j*h
        choi_minus = generate_choi(X_model)
        state_minus = apply_map(state_input, choi_minus)
        fid_minus = state_fidelity(state_minus, state_target)
        X_model[i, j] += 1j*h

        grad = 1j*(fid_plus-fid_minus)/h
        gradient_vector[j] += grad

    return gradient_vector


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

    def train(self, num_iter, use_adam=False):

        for step in tqdm(range(num_iter)):
            index = np.random.randint(0, 6**self.n-1)
            self.state_input = self.state_input_list[index]
            self.state_target = self.state_input_list[index]
            input_list = [(self, j) for j in range(self.rank)]

            # Finite difference over all parameters
            with mp.Pool(2) as pool:
                grad_matrix = np.array(pool.map(calculate_gradient, input_list)).T

            if use_adam:
                grad_matrix = self.adam(grad_matrix)

            self.X_model += self.lr*grad_matrix

            choi_model = generate_choi(self.X_model)
            state_model = apply_map(self.state_input, choi_model)
            fid = state_fidelity(state_model, self.state_target)

            self.fid_list.append(fid)
            print(f"{step}: {fid:.3f}")
