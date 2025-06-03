# SPAM Benchmarks

Here are notes regarding the SPAM benchmarks on experimental and synthetic data.

## SPAM Convergence

We investigate how well the SPAM models converge to the true SPAM model as a function of initialization and number of training steps. During training, we monitor the fidelity between the model and ground truth inital state and POVM using state fidelity and POVM fidelity, respectively. This lets us clearly see any conclusive signs of overfitting, as this circumvents possible weaknesses of using validation loss as a proxy for overfitting.

We also train multiple spam models on the same ground truth, but with different starting points. This lets us see if the model is able to robustly find the same solution, or if it gets stuck in local minima. This could be particularly interesting to invetigate for the corruption matrix model, since it is more constrained than the general POVM and might get stuck in local minima more easily.

### Results

# Experimental Data Spam Model

An essential part of testing and verifying our pipeline for recovering models of SPAM errors and quantum processes is to test it against synthetic data. Ideally, the synthetic data should be as realistic and similar to the dynamics of the real system as possible. This is a hard problem, since the real dynamics are unknown and are the thing we are interested in modelling in the first place.

Here we seek to provide a useful model for producing syntheic SPAM errors. We introduce the following SPAM model defined by

$\rho_{\text{True}} = (1-c_1)\rho_0 +  c_1 \rho(\boldsymbol{\omega})$
$E_{j,\text{True}} = (1-c_2)E_{j} +  c_2 E_{j}(\boldsymbol{\omega}),$

where $\rho_0$ and $E_{j}$ are correspondingly completely random versions parameterized by randomly sampled paramters $\omega$. $c_1$ and $c_2$ define a convex combination that tunes how much the SPAM model deviates from the ideal version, i.e., how much corruption is introduced.

## Setup

We gather Pauli String measurements from IBM Belem Quantum Computer as a basis for retrieving models of SPAM error. To get insight of the degree of corruption the SPAM errors of this particular Quantum Computer introduce, we compare the retrieved model with the ideal SPAM model using state fidelity for the initial state $\rho_0$, and povm fidelity for the POVM elements $E_j$.

Equivalently, we can compare the synthetic SPAM model with the ideal SPAM model. Ultimatly, $c_1$ and $c_2$ can be tuned such that the synthetic SPAM model corrupts to the same degree as the experimental SPAM model. This model is a natural way of implementing a completely random perturbation on the ideal SPAM model with distinct quantum mechanical features. While this is model may fail to recreate the particular details of real hardware, we see it as a still relavant and challinging benchmark to test our procedure for retrieving SPAM models.
