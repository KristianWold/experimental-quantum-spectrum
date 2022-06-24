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


def kron(*args):
    length = len(args)
    A = args[0]
    for i in range(1, length):
        A = torch.kron(A, args[i])

    return A

def pauli_observable(config):
    I = torch.eye(2)
    X = torch.tensor([[0, 1], [1, 0]])
    Y = torch.tensor([[0, -1j], [1j, 0]])
    Z = torch.tensor([[1, 0], [0, -1]])

    basis = [X, Y, Z]

    string = [basis[i] for i in config]
    observable = kron(*string).type(torch.complex128)

    return observable


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
    std = 1
    A = np.random.normal(0, std, (dim1, dim2))
    A = torch.from_numpy(A).type(torch.complex128)
    B = np.random.normal(0, std, (dim1, dim2))
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


def generate_unitary(ginibre):
    Q, R = torch.linalg.qr(ginibre, mode="reduced")
    U = Q@torch.diag(torch.sgn(torch.diag(R)))

    return U


def kraus_to_choi(kraus_map):
    d = kraus_map.kraus_list[0].shape[0]
    choi = torch.zeros((d**2, d**2), dtype=torch.complex128)
    for i in range(d):
        for j in range(d):
            M = torch.zeros((d,d), dtype=torch.complex128)
            M[i,j] = 1
            M_prime = kraus_map.apply_map(M)
            choi += torch.kron(M_prime, M)
    choi /= d

    return choi


def expectation_value(state, observable):
    ev = torch.trace(observable@state)
    return ev


def state_density_loss(model, input, target):
    state = input
    output = model.apply_map(input)
    loss = -state_fidelity(output, target)
    return loss


def expectation_value_loss(model, input, target):
    state, observable = input
    state = model.apply_map(state)
    output = expectation_value(state, observable, model)
    loss = (output - target)**2
    return loss


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

        state = [K@state@K.T.conj() for K in self.kraus_list]
        state = torch.stack(state, dim=0).sum(dim=0)
        return state

    def generate_map(self):
        d = self.d
        X = self.A + 1j*self.B
        U = generate_unitary(X)
        c = 1/(1 + torch.exp(-self.k))

        self.kraus_list = [(1-c)*U[i*d:(i+1)*d, :d] for i in range(self.rank)]
        self.kraus_list.append(c*self.U)


class ModelQuantumMap:

    def __init__(self, model, loss, input_list, target_list, lr):
        self.model = model
        self.loss = loss
        self.input_list = input_list
        self.target_list = target_list

        self.d = model.d
        self.rank = model.rank

        self.optim = optim.Adam(model.parameter_list, lr=lr)
        self.fid_list = []

#    @profile
    def train(self, num_iter):

        for step in tqdm(range(num_iter)):
            index = np.random.randint(len(self.input_list))
            input = self.input_list[index]
            target = self.target_list[index]

            self.optim.zero_grad()
            self.model.generate_map()
            loss = self.loss(self.model, input, target)
            loss.backward()
            self.optim.step()
            choi_model = kraus_to_choi(self.model)

            print(f"{step}: loss: {loss.detach().numpy():.3f}")
