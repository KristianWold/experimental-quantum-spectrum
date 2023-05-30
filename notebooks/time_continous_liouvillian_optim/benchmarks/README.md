# Benchmarks

## Precision Benchmarks

We investigate how the precision of our implemented Magnus Propagtor behaves as a function of number of grid points when integrating the Spin-Spin Hamiltonian.

A priori, the required time resolution $\delta t$ should correlate to how fast the time dependent Liouvillian (Hamiltonian part) fluctuates in time.
Since the Spin-Spin Hamiltonian have Fourier Series type terms, the characteristic fluctuation times should be determinded by how high degree terms are included. 
If $d$ is the highest degree term added, the typical time resolution to resolve the smallest fluctuations is given by half its periode $T/2 = \frac{2\pi}{4\pi d} = \frac{1}{2d}$

Since the Spin-Spin Hamiltonian is not exactly solvable, we generate a ground truth solution by running the Magnus Propagator for a high number of grid points compared to the fluctuations of the Hamiltonian ($n = 10000$). Then, we recompute the solution with gradually fewer gridpoints, and check for consistency with the ground truth.  

## Robustness Benchmarks

Optimization, especially gradient based, is sometimes prone to getting stuck in local minima in the loss landscape. Worst case, these local minima are bad, and represent solutions far away from the ideal global minimum. 



