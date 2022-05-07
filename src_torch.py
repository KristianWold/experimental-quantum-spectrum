import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch

from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm
from sqrtm import sqrtm


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


def bitstring_density(state, basis):

    return np.abs(np.diag(basis.conj().T@state@basis))


def generate_ginibre(dim1, dim2, real = False):
    ginibre = np.random.normal(0, 1, (dim1, dim2))
    if not real:
         ginibre = ginibre + 1j*np.random.normal(0, 1, (dim1, dim2))
    return torch.from_numpy(ginibre).type(torch.complex128)


def generate_state(dim1, dim2):
    X = generate_ginibre(dim1, dim2)

    state = X@X.conj().T/torch.trace(X@X.conj().T)
    return state

def square_root_inverse(A):
    A = sqrtm(A)
    A = torch.inverse(A).contiguous()
    
    return A

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

        num_workers = min(self.rank, mp.cpu_count()//2)

        for step in tqdm(range(num_iter)):
            index = np.random.randint(0, len(self.state_input_list)-1)
            self.state_input = self.state_input_list[index]
            self.state_target = self.state_target_list[index]

            choi_model = generate_choi(self.X_model)
            state_model = apply_map(self.state_input, choi_model)
            fid = state_fidelity(state_model, self.state_target)

            self.fid_list.append(fid)
            print(f"{step}: {fid:.3f}")
