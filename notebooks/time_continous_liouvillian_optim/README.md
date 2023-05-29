# Optimizing Parametric Time-Continuous Qubit Liovillians

We implement a general TensorFlow framework for simulating and optimizing a large family of Time-Continuous Qubit Liovillians.
The Liovillians are implemented as differentiable functions dependent on parameters. Interesting Liovillians may be discovered by learning the paramters of the model by minimizing a cost function over some data.

## Hamiltonian and Dissipative Part, Ideal Quantum Gate Target

We will investigate Liovillians with a distinct parametric Hamiltonian part and constant dissipative part. In the perspective of this project, the Hamiltonian can reflect capabilities and details possible to implement in quantum hardware, such as time dependent manipulation of microwave, magnetic or electric pulses. The dissipative represent sources of decoherence in the quantum hardware induced by the environment. This dissipative part will be assumes, slightly unrealistically, to be time independent and unrelated to the Hamiltonian description.

## Spin-Spin Hamiltonian

The first model Hamiltonian we study is a two-qubit Hamiltonian, with Fourier expanded arbitrary pulses local to each qubit, together with a static and anisotropic spin-spin coupling between them. This Hamiltonian can be expressed as

$H(t;\theta, \omega, \gamma) = \sum_{j=1}^3\sum_{k=0}^d \sum_{l=1}^2 \theta_{jk}^{(l)} \sin(2\pi kt) \sigma_j^{(l)} + \omega_{jk}^{(l)} \cos(2\pi kt)\sigma_j^{(l)} +\sum_{j=1}^3\gamma_j \sigma_j\otimes \sigma_j,$

where $\sigma_j^{(1)} = \sigma_j \otimes I$ and $\sigma_j^{(2)} =I \otimes \sigma_j$. $d$ gives the degree of coefficients (in terms of $sin$ and $cos$ terms) to include in the expansion of the pulses. These are counted individually for each qubit, for each of the three Pauli couplings $X$, $Y$ and $Z$. Each of the terms are weighted with learnable coefficients $\theta_{jk}^{(l)}$ and $\omega_{jk}^{(l)}$($\sin$ and $\cos$, respectivly), where $l$ indicates which qubit is interacted with, $j$ what Pauli coupling, and $k$ the degree degree. Note that for $k=0$, the contribution of $\sin$ terms vanishes. These non-contributing terms are neglected in the implementation, but are included mathematically here for easier notation.

This formualtion gives very generic control over the each qubit as sufficiently large degree $d$ lets one approximate any single-qubit dynamic evolution to an arbitrary precision on the interval t=0 to t=1. However, the coupling of the qubits are a time-independent feature, so it is not a priori obvious whether or not this Hamiltonian familty can approximate any two-qubit dynamics.

### Quantum Gate from Spin-Spin Hamiltonian

We want to find appropriate parameters for the Spin-Spin Hamiltonian such that interesting Quantum Gates, that is, unitary dynamics, is realized. The Hamiltonian relates to the resulting unitary in the following way:

$U(t) = \mathcal{T}e^{-i \int_0^t H(t)dt'}$.


