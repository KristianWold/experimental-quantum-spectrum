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

from quantum_tools import *
from spam import *
from utils import *
from set_precision import *


def reshuffle(A):
    d = int(np.sqrt(A.shape[0]))
    A = tf.reshape(A, (d,d,d,d))
    A = tf.einsum("jklm -> jlkm", A)
    A = tf.reshape(A, (d**2,d**2))

    return A


def kraus_to_choi(kraus_channel, use_reshuffle = True):
    kraus = kraus_channel.kraus
    rank = kraus.shape[1]
    channel = 0

    for i in range(rank):
        K = kraus[0, i]
        channel += tf.experimental.numpy.kron(K, tf.math.conj(K))

    if use_reshuffle:
        choi = reshuffle(channel)

    return choi

def state_purity(A):
    eig, _ = tf.linalg.eig(A)
    purity = tf.math.reduce_sum(eig**2)
    return purity


def effective_rank(channel):
    choi = channel.choi
    d2 = choi.shape[0]
    
    purity = state_purity(choi)
    
    rank_eff = d2/purity
    return rank_eff


def channel_to_choi(channel_list):
    if not isinstance(channel, list):
        channel_list = [channel_list]

    d = channel_list[0].d
    choi = tf.zeros((d**2, d**2), dtype=precision)
    M = np.zeros((d**2, d, d))
    for i in range(d):
        for j in range(d):

            M[d*i + j, i, j] = 1

    M = tf.convert_to_tensor(M, dtype=precision)
    M_prime = tf.identity(M)
    for channel in channel_list:
        M_prime = channel.apply_channel(M_prime)
    for i in range(d**2):
        choi += tf.experimental.numpy.kron(M_prime[i], M[i])

    return choi


def channel_fidelity(channel_A, channel_B):
    choi_A = kraus_to_choi(channel_A)
    choi_B = kraus_to_choi(channel_B)
    d_squared = choi_A.shape[0]
    fidelity = state_fidelity(choi_A, choi_B)/d_squared

    return fidelity


def channel_spectrum(channel, real=True):
    eig, _ = tf.linalg.eig(reshuffle(channel.choi))
    eig = tf.expand_dims(eig, axis=1)
    
    if real:
        x = tf.cast(tf.math.real(eig), dtype=precision)
        y = tf.cast(tf.math.imag(eig), dtype=precision)
        eig = tf.concat([x, y], axis=1)

    return eig


def kraus_spectrum(kraus):
    rank = kraus.shape[1]
    d = kraus.shape[2]
    vectors = tf.reshape(kraus, (rank, d**2))
    print(vectors.shape)
    norm = tf.linalg.adjoint(vectors)*vectors
    norm = tf.math.sqrt(tf.math.reduce_sum(norm, axis=1))
    
    return norm


def normalize_spectrum(spectrum):
    spectrum = spectrum.numpy()
    idx = np.argmax(np.linalg.norm(spectrum, axis=1))
    spectrum[idx] = (0,0)

    max = np.max(np.linalg.norm(spectrum, axis=1))
    print(max)
    spectrum = 1/max*spectrum

    spectrum[idx] = (1,0)
    spectrum = tf.cast(tf.convert_to_tensor(spectrum), dtype=precision)

    return spectrum

def choi_steady_state(choi):
    d = int(np.sqrt(choi.shape[0]))
    choi = reshuffle_choi(choi)
    eig, eig_vec = np.linalg.eig(choi)
    steady_index = tf.math.argmax(tf.abs(eig))

    steady_state = eig_vec[:, steady_index]
    steady_state = steady_state.reshape(d, d)
    steady_state = steady_state/tf.linalg.trace(steady_state)

    return steady_state


def dilute_channel(U, c, kraus_map):
    pass


class Channel():
    def __init__(self, d, rank, spam):
        d = None,
        rank = None,
        spam = None

    def generate_channel(self):
        pass

    def apply_channel(self, state):
        pass

    @property    
    def choi(self):
        pass


class KrausMap(Channel):

    def __init__(self,
                 d = None,
                 rank = None,
                 spam = None,
                 trainable = True,
                 generate = True,
                 ):

        self.d = d
        self.rank = rank

        if spam is None:
            spam = SPAM(d=d,
                        init = init_ideal(d),
                        povm = povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(rank*d, d, trainable = trainable)
        self.parameter_list = [self.A, self.B]

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = tf.cast(self.A, dtype=precision) + 1j*tf.cast(self.B, dtype=precision)
        U = generate_unitary(G=G)
        self.kraus = tf.reshape(U, (1, self.rank, self.d, self.d))


    def apply_channel(self, state):
        state = tf.expand_dims(state, axis=1)
        Kstate = tf.matmul(self.kraus, state)
        KstateK = tf.matmul(Kstate, self.kraus, adjoint_b=True)
        state = tf.reduce_sum(KstateK, axis=1)

        return state

    @property    
    def choi(self):
        return kraus_to_choi(self)


class DilutedKrausMap(KrausMap):

    def __init__(self,
                 U = None,
                 c = None,
                 d = None,
                 rank = None,
                 spam = None,
                 trainable = True,
                 generate = True,
                 ):

        KrausMap.__init__(d, rank, spam, trainable, generate=False)

        self.U = U
        if self.U is not None:
            self.U = tf.expand_dims(tf.expand_dims(self.U, 0), 0)
            k = -np.log(1/c - 1)
            self.k = tf.Variable(tf.cast(k, dtype = precision), trainable = True)
            self.parameter_list.append(self.k)
        else:
            self.k = None

        self.kraus = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        Kraus.generate_channel()
        
        if self.U is not None:
            c = 1/(1 + tf.exp(-self.k))
            self.kraus = tf.concat([tf.sqrt(c)*self.U, tf.sqrt(1-c)*self.kraus], axis=1)

    @property
    def c(self):
        if self.k is None:
            c = None
        else:
            c = 1/(1 + tf.exp(-self.k))
        return c


class SquaredKrausMap(KrausMap):
    def __init__(self,
                d = None,
                rank = None,
                spam = None,
                trainable = True,
                generate = True,
                ):

            KrausMap.__init__(d, rank, spam, trainable, generate=generate)


    def apply_channel(self, state):
        state = KrausMap.apply_channel(state)
        state = KrausMap.apply_channel(state)

        return state

    @property    
    def choi(self):
        return channels_to_choi(self)


class ChoiMap():

    def __init__(self,
                 d = None,
                 rank = None,
                 spam = None,
                 trainable = True,
                 generate = True,
                 ):

        self.d = d
        self.rank = rank
        self.I = tf.cast(tf.eye(self.d), dtype = precision)

        if spam is None:
            spam = SPAM(d=d,
                        init = init_ideal(d),
                        povm = povm_ideal(d))
        self.spam = spam

        _, self.A, self.B = generate_ginibre(d**2, rank, trainable = trainable)
        self.parameter_list = [self.A, self.B]

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = self.A + 1j*self.B
        
        XX = tf.matmul(G, G, adjoint_b = True)
        
        Y = partial_trace(XX)
        Y = tf.linalg.sqrtm(Y)
        Y = tf.linalg.inv(Y)
        Ykron = kron(self.I, Y)

        choi = Ykron@XX@Ykron
        self.super_operator = reshuffle(choi)

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    @property    
    def choi(self):
        return reshuffle(self.super_operator)


class LindbladMap():

    def __init__(self,
                 d = None,
                 rank = None,
                 spam = None,
                 trainable = True,
                 generate = True,
                 ):

        self.d = d
        self.rank = rank
        self.I = tf.cast(tf.eye(d), dtype = precision)

        if spam is None:
            spam = SPAM(d=d,
                        init = init_ideal(d),
                        povm = povm_ideal(d))
        self.spam = spam

        _, A, B = generate_ginibre(d, d, trainable = trainable)
        self.H_params = [A, B]
        _, A, _ = generate_ginibre(rank-1, 1, trainable = trainable, complex=False)
        self.gamma_params = [A]

        self.A_list = []
        self.B_list = []
        for i in range(rank-1):
            _, A, B = generate_ginibre(d, d, trainable = trainable)
            self.A_list.append(A)
            self.B_list.append(B)

        self.parameter_list = self.H_params + self.gamma_params + self.A_list + self.B_list

        self.super_operator = None
        if generate:
            self.generate_channel()

    def generate_channel(self):
        G = self.H_params[0] +1j*self.H_params[0]
        H = G + tf.linalg.adjoint(G)
        gamma = tf.cast(tf.abs(self.gamma_params[0]), dtype = precision)

        L_list = [A + 1j*B for A,B in zip(self.A_list, self.B_list)]
        
        ab = tf.linalg.trace(tf.matmul(self.I/np.sqrt(self.d), L_list[0], adjoint_a=True))
        L_list[0] = (L_list[0] -  ab*self.I/np.sqrt(self.d))
        L_list[0] = L_list[0]/tf.math.sqrt(tf.linalg.trace(tf.matmul(L_list[0], L_list[0], adjoint_a=True)))

        for i in range(1, len(L_list)):
            for j in range(i):
                ab = tf.linalg.trace(tf.matmul(L_list[j], L_list[i], adjoint_a=True))
                L_list[i] += -ab*L_list[j]

            L_list[i] = L_list[i]/tf.math.sqrt(tf.linalg.trace(tf.matmul(L_list[i], L_list[i], adjoint_a=True)))
        
        LB = -1j*(kron(H, self.I) - kron(self.I, tf.math.conj(H)))
        for i, L in enumerate(L_list):
            L2 = tf.matmul(L, L, adjoint_a=True)
            LB += gamma[i]*(kron(L, tf.math.conj(L)) - 0.5*((kron(L2, self.I) + kron(self.I, tf.math.conj(L2)))))
        
        self.super_operator = tf.linalg.expm(LB)
        

    def apply_channel(self, state):
        state = tf.reshape(state, (-1, d**2, 1))
        state = tf.matmul(self.super_operator, state)
        state = tf.reshape(state, (-1, d, d))

        return state

    @property    
    def choi(self):
        return reshuffle(self.super_operator)



