import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch
import torch.optim as optim

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


def partial_trace(X, discard_first = True):
    d = int(np.sqrt(X.shape[0]))
    X = X.reshape(d,d,d,d)
    if discard_first:
        Y = torch.einsum("ijik->jk", X)
    else:
        Y = torch.einsum("jiki->jk", X)
    return Y


def state_fidelity(A, B):
    sqrtB = square_root(B)
    fidelity = torch.trace(square_root(sqrtB@A@sqrtB))
    return fidelity


def apply_map(state, choi):
    d = state.shape[0]

    #reshuffle
    choi = choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)

    #flatten
    state = state.reshape(-1, 1)

    state = (choi@state).reshape(d, d)
    return state

def apply_kraus(state, kraus_list):
    state = [K.T.conj()@state@K for K in kraus_list]
    state = torch.stack(state, dim=0).sum(dim=0)

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
        state = torch.from_numpy(DensityMatrix(circuit).data).type(torch.complex128)
    else:
        state = Operator(circuit).data

    return state


def generate_ginibre(dim1, dim2, requires_grad=False):
    A = np.random.normal(0, 1, (dim1, dim2))
    A = torch.from_numpy(A).type(torch.complex128)
    B = np.random.normal(0, 1, (dim1, dim2))
    B = torch.from_numpy(B).type(torch.complex128)
    if requires_grad:
        ginibre = A.requires_grad_() + 1j*B.requires_grad_()
    else:
        ginibre = A + 1j*B
    return ginibre, A, B


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/torch.trace(X@X.conj().T)
    return state

def square_root_inverse(A):
    L, V = torch.linalg.eig(A)
    L = 1/torch.sqrt(L)

    B = torch.zeros_like(A)
    for l, v in zip(L, V.T):
        B += l*torch.conj(v.reshape(-1,1))@v.reshape(1,-1)

    return B

def square_root(A):
    L, V = torch.linalg.eig(A)
    L = torch.sqrt(L)

    B = torch.zeros_like(A)
    for l, v in zip(L, V.T):
        B += l*torch.conj(v.reshape(-1,1))@v.reshape(1,-1)

    return B


def generate_choi(X):
    d = int(np.sqrt(X.shape[0]))  # dim of Hilbert space
    I = torch.eye(d).type(torch.complex128)
    XX = X@X.T.conj()

    #partial trace
    Y = square_root_inverse(partial_trace(XX, discard_first=True))
    Ykron = torch.kron(I, Y).T

    #choi
    choi = Ykron@XX@Ykron

    return choi


def generate_unitary(ginibre):
    Q, R = torch.linalg.qr(ginibre)
    U = Q@torch.diag(torch.sgn(torch.diag(R)))

    return U


class ChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        self.ginibre = generate_ginibre(d**2, rank)
        self.parameter_list = [self.ginibre]

        self.choi = None
        self.generate_map()
        self.k = np.array([-10], dtype = "float64")

    def apply_map(self, state):
        d = self.d

        #reshuffle
        choi = self.choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)

        #flatten
        state = state.reshape(-1, 1)

        state = (choi@state).reshape(d, d)
        return state

    def generate_map(self):
        I = np.eye(self.d)
        X = self.ginibre
        XX = X@X.conj().T
        #partial trace
        Y = partial_trace(XX)
        Y = sqrtm(Y)
        Y = np.linalg.inv(Y)
        Ykron = np.kron(I, Y)

        #choi
        self.choi = Ykron@XX@Ykron

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter += weight_gradient

        self.generate_map()


class KrausMap():

    def __init__(self, U, c, d, rank, requires_grad=False):
        self.U = U
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(rank*d, d, requires_grad=requires_grad)
        k = -np.log(1/c - 1)
        self.k = torch.tensor(k)
        if requires_grad:
            self.k = self.k.requires_grad_()

        self.parameter_list = [self.A, self.B, self.k]

        self.kraus_list = None
        self.generate_map()

    def apply_map(self, state):
        c = 1/(1 + torch.exp(-self.k))
        state = [c*self.U@state@self.U.T.conj()] + [(1 - c)*K@state@K.T.conj() for K in self.kraus_list]
        state = torch.stack(state, dim=0).sum(dim=0)
        return state

    def generate_map(self):
        d = self.d
        X = self.A + 1j*self.B
        U = generate_unitary(X)
        self.kraus_list = [U[i*d:(i+1)*d, :d] for i in range(self.rank)]


class ModelQuantumMap:

    def __init__(self, model, state_input_list, state_target_list, lr):
        self.model = model
        self.state_input_list = state_input_list
        self.state_target_list = state_target_list

        self.d = model.d
        self.rank = model.rank

        self.optim = optim.Adam(model.parameter_list, lr=lr)
        self.fid_list = []

#    @profile
    def train(self, num_iter):

        for step in tqdm(range(num_iter)):
            index = np.random.randint(len(self.state_input_list))
            self.state_input = self.state_input_list[index]
            self.state_target = self.state_target_list[index]

            self.optim.zero_grad()
            self.model.generate_map()
            state_model = self.model.apply_map(self.state_input)
            loss = -state_fidelity(self.state_target, state_model)
            loss = torch.norm(self.state_target - state_model)
            loss.backward()
            self.optim.step()

            print(f"{step}: loss: {loss.detach().numpy():.3f}")
