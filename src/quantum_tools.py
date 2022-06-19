import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm


def state_fidelity(A, B):

    sqrtB = sqrtm(B)
    C = sqrtB@A@sqrtB

    fidelity = np.trace(sqrtm(C))
    return np.abs(fidelity)**2


def state_norm(A, B):

    norm = 1 - np.linalg.norm(A - B)
    return np.abs(norm)


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

def expectation_value(state, observable):
    ev = np.trace(observable@state)
    return ev


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


def kron(*args):
    length = len(args)
    A = args[0]
    for i in range(1, length):
        A = np.kron(A, args[i])

    return A


def pauli_observable(config, return_mode = "density"):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    basis = [I, X, Y, Z]

    if return_mode == "density":
        string = [basis[idx] for idx in config]
        result = kron(*string)

    if return_mode == "circuit":
        n_sub = sum([idx != 0 for idx in config])
        n_count = 0

        q_reg = qk.QuantumRegister(len(config))
        c_reg = qk.ClassicalRegister(n_sub)
        circuit = qk.QuantumCircuit(q_reg, c_reg)

        for i, index in enumerate(config):
            if index == 1:
                circuit.h(i)

            if index == 2:
                circuit.rz(-np.pi/2, i)    
                circuit.h(i)

            if index == 3:
                pass    #measure in computational basis

            if index != 0:
                circuit.measure(q_reg[i], c_reg[n_count])
                n_count += 1

        result = circuit

    return result


#@profile
def generate_ginibre(dim1, dim2):
    A = np.random.normal(0, 1, (dim1, dim2))
    B = np.random.normal(0, 1, (dim1, dim2))
    X = A + 1j*B
    return X, A, B


def generate_state(d, rank):
    X, _, _ = generate_ginibre(d, rank)

    state = X@X.conj().T/np.trace(X@X.conj().T)
    return state


def generate_unitary(X):
    Q, R = np.linalg.qr(X)
    R = np.diag(R)
    sign = R/np.abs(R)
    U = Q@np.diag(sign)

    return U


def reshuffle_choi(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = choi.reshape(d,d,d,d).swapaxes(1,2).reshape(d**2, d**2)
    return choi


def choi_spectrum(choi):
    choi = reshuffle_choi(choi)
    eig, _ = np.linalg.eig(choi)

    x = np.real(eig)
    y = np.imag(eig)

    return np.array([x, y])
