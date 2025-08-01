# Analysis

This folder contains all notebooks and data required to regenerate the results and figures in the paper.

## Atypical Quantum Maps
Experimentally realistic synthetic data is generated on quantum maps with known spesification. They are retrived using our quantum process tomography protocol and shown to be consistent with the targets on the spectral level without significant bias or error.

Reproduces figure 7.

# Concatenation
Quantum maps are retrieved on real experimental data, stemming from parameterized quantum circuits, and concatenation of such circuits. These circuits were ran on the IBM Belem quantum computers. We show that our retrieved maps predict unseen Pauli string expectation values better than the ideal unitary circuit descriptions.

Reproduces figure 2.

## Diluted Unitary
Quantum maps are retrieved on real experimental data, stemming from parameterized quantum circuits.

Reproduces figure 3 and 8.

## Experiments

Parameterized Quantum Ciruits are constructed using python module Qiskit. These are then repeatedly executed on the IBM Belen quantum computer for different Pauli strings, using the IBM Quantum experience API. 

## Expressivity

Simulation is used to show that the (ideal) parameterized quantum circuits used in this work results in measurement values indistinguishable from Haar random unitaries. In this sense, the experimental implementation detailed in Experiments represents a noisy but otherwise random unitary realization. 

Reproduces figure 5.

## SPAM benchmarks

Experimentally realistic synthetic data is generated using known state preparation and measurement (SPAM) error models. They are retrived using our SPAM tomography protocol and shown to be consistent with the targets without significant bias or error.

Reproduces figure 6.