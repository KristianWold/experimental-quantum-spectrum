# Precision and Time Benchmarks

We investigate how the precision of our implemented Magnus Propagtor.

A priori, the required resolution of time should correlate to how fast the time dependent Liouvillian (Hamiltonian part) fluctuates in time.
Since the Spin-Spin Hamiltonian have Fourier Series type terms, the characteristic fluctuation times should be determinded by how high degree terms are included. 
If $d$ is the highest degree term added, the typical time resolution to resolve the smallest fluctuations is given by half its periode $T/2 = \frac{2\pi}{4\pi d} = \frac{1}{2d}$  