import numpy as np
import qiskit as qk
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import Operator
from scipy.linalg import sqrtm
from tqdm.notebook import tqdm

from src_tf_noBP import *

n = 3
d = 2**n

state_input_list = [prepare_input(numberToBase(i, 6, n)) for i in range(6**n)]

np.random.seed(42)
X_target = generate_ginibre(d**2, 2)
choi_target = generate_choi(X_target)
state_target_list = [apply_map(state_input, choi_target) for state_input in state_input_list]

model2 = ModelQuantumMap(n = 3,
                         rank = 2,
                         state_input_list = state_input_list,
                         state_target_list = state_target_list,
                         lr = 0.05,
                         h = 1e-4)

model2.train(num_iter = 20,
             use_adam = True)
