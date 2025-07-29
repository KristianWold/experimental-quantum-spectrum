# Source Code

This is a custom code base, written mainly with TensorFlow, qiskit, and NumPy, for modelleling quantum maps from experimental data and analyzing resulting quantum spectra. 

## experimental.py

Helper functions for generating and running quantum circuits with IBM's Quantum Cloud API, in particular setting up Pauli strings and setting experimental parameters.

## kraus_channels.py

Classes for generating parameterized and differentiable unitaries, Kraus maps, and Diluted Unitaries.

## loss_functions.py

Various loss function used as optimizations criterions in this project.

## optimization.py

Classes and helper functions for optimizing quantum map models and SPAM models using TensorFlows built-in autodiff.

## quantum_channels.py

Various functions for treating and analysing quantum maps.

## quantum circuits.py

Functions for creating quantum circuits descriptions, like the parameterized quantum circuit (PQC).

## set_precision.py

Global parameter for model precision, assued to be complex128.

## spam.py

Classes and functions for creating SPAM models.

## synthetic_data.py

Functions for generating experimentally realistic synthetic data, used for benchmarking.

## utils.py

Various helper functions, like saver and loader functions.