import numpy as np
import qiskit as qk
import tensorflow as tf

from tqdm.notebook import tqdm
from utils import *
from set_precision import *
from quantum_tools import *
from experimental import *


def angular_histogram(spectrum_list, bins, color="b"):
    angular_list = [spectrum_to_angular(spectrum) for spectrum in spectrum_list]
    angular = np.concatenate(angular_list)
    plt.hist(angular, bins, color=color)


def find_outer_inner_R(spectrum_list, tail_num=10):
    if isinstance(spectrum_list, list):
        radial_list = np.array(
            [np.sort(spectrum_to_radial(spectrum)) for spectrum in spectrum_list]
        )
        R_minus = np.mean(radial_list[:, 0])
        R_minus_std = np.std(radial_list[:, 0])
        R_plus = np.mean(radial_list[:, -1])
        R_plus_std = np.std(radial_list[:, -1])
    else:
        radial_list = np.sort(spectrum_to_radial(spectrum_list))
        R_minus = np.mean(radial_list[:tail_num])
        R_minus_std = np.std(radial_list[:tail_num])
        R_plus = np.mean(radial_list[-tail_num:])
        R_plus_std = np.std(radial_list[-tail_num:])

    return R_plus, R_minus, R_plus_std, R_minus_std


def annulus_distance(spectrum1, spectrum2):
    angular1 = spectrum_to_angular(spectrum1)
    a_mean1 = np.mean(angular1)
    a_std1 = np.std(angular1)

    angular2 = spectrum_to_angular(spectrum2)
    a_mean2 = np.mean(angular2)
    a_std2 = np.std(angular2)

    radial1 = spectrum_to_radial(spectrum1)
    r_mean1 = np.mean(radial1)
    r_std1 = np.std(radial1)

    radial2 = spectrum_to_radial(spectrum2)
    r_mean2 = np.mean(radial2)
    r_std2 = np.std(radial2)

    distance = (
        np.abs(r_mean1 - r_mean2)
        + np.abs(r_std1 - r_std2)
        + np.abs(a_mean1 - a_mean2)
        + np.abs(a_std1 - a_std2)
    )

    return distance
