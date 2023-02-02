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
from qiskit.circuit.library import iSwapGate



def pqc_basic(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 2 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n - 1):
            circuit.cx(i, i + 1)

    return circuit


def pqc_expressive(n, L):
    theta_list = [np.random.uniform(0, 2 * np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)

        for i in range(n):
            circuit.crx(theta[i + n], i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)

        for i in reversed(list(range(n))):
            circuit.crx(theta[3 * n + i], (i + 1) % n, i)

    return circuit


def pqc_more_expressive(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 4 * n) for i in range(L)]
    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.ry(theta[i], i)
            circuit.rz(theta[i + n], i)

        for i in range(n):
            circuit.cx(i, (i + 1) % n)

        for i in range(n):
            circuit.ry(theta[i + 2 * n], i)
            circuit.rx(theta[i + 3 * n], i)

        for i in range(n):
            circuit.cx(n - i - 1, n - (i + 1) % n - 1)

    return circuit


def integrabel_clifford(n, L):
    theta_list = [np.random.uniform(-np.pi, np.pi, 3 * n) for i in range(L)]
    sqrt_iSWAP = iSwapGate().power(1/2)

    circuit = qk.QuantumCircuit(n)
    for theta in theta_list:
        for i in range(n):
            circuit.rz(theta[i], i)

        for i in range(n//2):
            circuit.append(sqrt_iSWAP, [2*i, 2*i+1])

        for i in range(n):
            circuit.rz(theta[n+i], i)

        for i in range((n-1)//2):
            circuit.append(sqrt_iSWAP, [2*i+1, 2*i+2])
        
        for i in range(n):
            circuit.rz(theta[2*n+i], i)

        for i in range(n//2):
            circuit.append(sqrt_iSWAP, [2*i, 2*i+1])

    return circuit

