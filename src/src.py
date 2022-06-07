import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
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
    return np.abs(fidelity)**2


def channel_fidelity(map_A, map_B):
    choi_A = maps_to_choi([map_A])
    choi_B = maps_to_choi([map_B])
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def state_norm(A, B):

    norm = 1 - np.linalg.norm(A - B)
    return np.abs(norm)


def maps_to_choi(map_list):
    d = map_list[0].d
    choi = np.zeros((d**2, d**2), dtype="complex128")
    for i in range(d):
        for j in range(d):
            M = np.zeros((d,d), dtype="complex128")
            M[i,j] = 1
            M_prime = np.copy(M)
            for map in map_list:
                M_prime = map.apply_map(M_prime)

            choi += np.kron(M_prime, M)

    return choi


def expectation_value(state, observable, q_map):
    state = q_map.apply_map(state)
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


def pauli_observable(config):
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    basis = [I, X, Y, Z]

    string = [basis[i] for i in config]
    observable = kron(*string)

    return observable


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


def state_density_loss(q_map, input, target, grad=False):
    state = input
    output = q_map.apply_map(input)
    cost = -state_fidelity(output, target)
    return cost


def expectation_value_loss(q_map, input, target, grad=False):
    state, observable = input
    state = q_map.apply_map(state)
    output = expectation_value(state, observable, q_map)
    cost = np.abs(output - target)**2
    return cost


def channel_fidelity_loss(q_map, input, target, grad=False):
    q_map_target = input
    cost = -channel_fidelity(q_map, q_map_target)
    return cost


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
        #reshuffle
        choi = reshuffle_choi(self.choi)

        #flatten
        state = state.reshape(-1, 1)

        state = (choi@state).reshape(self.d, self.d)
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter -= weight_gradient

        self.generate_map()


class DoubleChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        self.double_choi = [ChoiMap(d, rank), ChoiMap(d, rank)]
        self.parameter_list = []
        self.parameter_list.extend(self.double_choi[0].parameter_list)
        self.parameter_list.extend(self.double_choi[1].parameter_list)


        self.generate_map()
        self.k = np.array([[-10]], dtype = "float64")

    def generate_map(self):
        self.double_choi[0].generate_map()
        self.double_choi[1].generate_map()

    def apply_map(self, state):
        #reshuffle
        state = self.double_choi[0].apply_map(state)
        state = self.double_choi[1].apply_map(state)
        return state

    def update_parameters(self, weight_gradient_list):
        self.double_choi[0].update_parameters(weight_gradient_list[:2])
        self.double_choi[1].update_parameters(weight_gradient_list[2:])


class SquareRootChoiMap():

    def __init__(self, d, rank):
        self.d = d
        self.rank = rank

        self.square_root_choi = ChoiMap(d, rank)
        self.parameter_list = self.square_root_choi.parameter_list


        self.generate_map()
        self.k = np.array([[-10]], dtype = "float64")

    def generate_map(self):
        self.square_root_choi.generate_map()

    def apply_map(self, state):
        #reshuffle
        state = self.square_root_choi.apply_map(self.square_root_choi.apply_map(state))
        return state

    def update_parameters(self, weight_gradient_list):
        self.square_root_choi.update_parameters(weight_gradient_list)


class KrausMap():

    def __init__(self,
                 U=None,
                 c=None,
                 d=None,
                 rank = None,
                 generate_map = True):
        self.U = U
        self.d = d
        self.rank = rank

        _, self.A, self.B = generate_ginibre(rank*d, d)
        self.parameter_list = [self.A, self.B]

        if self.U is not None:
            k = -np.log(1/c - 1)
            self.k = np.array([[k]], dtype = "float64")
            self.parameter_list.append(self.k)

        self.kraus_list = None
        if generate_map:
            self.generate_map()

    def generate_map(self, U=None):
        d = self.d
        X = self.A + 1j*self.B
        if U is None:
            U = generate_unitary(X)

        self.kraus_list = []
        if self.U is not None:
            c = 1/(1 + np.exp(-self.k[0,0]))
            self.kraus_list.append(np.sqrt(c)*self.U)
        else:
            c = 0

        self.kraus_list.extend([np.sqrt(1-c)*U[i*d:(i+1)*d, :d] for i in range(self.rank)])

    def apply_map(self, state):

        state = sum([K@state@K.T.conj() for K in self.kraus_list])
        return state

    def update_parameters(self, weight_gradient_list):
        for parameter, weight_gradient in zip(self.parameter_list, weight_gradient_list):
            parameter -= weight_gradient

        self.generate_map()


class ModelQuantumMap:

    def __init__(self, q_map, cost, input_list, target_list, lr, h):
        self.q_map = q_map
        self.cost = cost
        self.input_list = input_list
        self.target_list = target_list
        self.lr = lr
        self.h = h

        self.d = q_map.d
        self.rank = q_map.rank

        self.adam = Adam()
        self.fid_list = []

#    @profile
    def train(self, num_iter, use_adam=False, verbose=True, N = 1, choi_target=None):

        self.cost_average = sum([self.cost(self.q_map, input, target) for input, target in zip(self.input_list, self.target_list)])/len(self.input_list)

        for step in tqdm(range(num_iter)):

            grad_list = [np.zeros_like(parameter) for parameter in self.q_map.parameter_list]
            for batch in range(N):
                index = np.random.randint(len(self.input_list))
                self.input = self.input_list[index]
                self.target = self.target_list[index]

                for parameter, grad in zip(self.q_map.parameter_list, grad_list):
                    grad_matrix = self.calculate_gradient(parameter)
                    grad += grad_matrix

            for grad in grad_list:
                grad /= N

            if use_adam:
                grad_list = self.adam(grad_list, self.lr)

            self.q_map.update_parameters(grad_list)

            self.cost_average = sum([self.cost(self.q_map, input, target) for input, target in zip(self.input_list, self.target_list)])/len(self.input_list)

            c = 1/(1 + np.exp(-self.q_map.k[0,0]))

            if choi_target is not None:
                choi_model = kraus_to_choi([self.q_map])
                fid = state_fidelity(choi_model, choi_target)
            else:
                fid = self.cost_average

            self.fid_list.append(fid)
            if verbose:
                print(f"{step}: fid: {fid:.3f}, c: {c:.3f}")

#    @profile
    def calculate_gradient(self, parameter):

        h = self.h
        input = self.input
        target = self.target

        grad_matrix = np.zeros_like(parameter)
        for i in range(parameter.shape[0]):
            for j in range(parameter.shape[1]):
                parameter[i, j] += h
                self.q_map.generate_map()
                cost_plus = self.cost(self.q_map, input, target, grad=True)

                parameter[i, j] -= 2*h
                self.q_map.generate_map()
                cost_minus = self.cost(self.q_map, input, target, grad=True)

                parameter[i, j] += h
                grad_matrix[i, j] = (cost_plus-cost_minus)/(2*h)

        return grad_matrix
