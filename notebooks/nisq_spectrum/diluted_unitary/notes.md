<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>

# Diluted Unitary Fitting

We explore the capabilities of so-called Diluted Unitary models for recreating statistical spectral properties of NISQ circuits.

Diluted Unitaries are quantum maps on the form

$T(\rho) = (1-p)U\rho U^{\dagger} + p \sum_{i=1}^r K_i \rho K_i^{\dagger}$, where $K_i$ randomly sampled Kraus operators by the method of blocking Haar random (semi-)unitary matrices. U is usually also Haar random, or some specific unitary circuit description. This leaves just two free parameters of the Diluted Unitary: A decoherence parameter $p$ and the Kraus rank $r$.

The quantum channels induced by NISQ circuits are ultimatly a result of the circuit description and the specific details of the hardware. It is not a priori obvious that any characteristics of the resulting spectra can be captured by a simple model like the Diluted Unitary

To quantify a notion of similarity between the NISQ circuit spectra and Diluted Unitaries, we introduce a measure that captures the course-grained features that are similar in both type of spectra. At a first glance, NISQ circuits and Diluted Unitaries exhibit annulus-shaped spectra with a well defined inner and out radius. The annulus can be defined in terms of inner radius and outer, or, equivalently, mean radius and standard deviation of radius, if one imagines it as a uniform mass object.

We introduce the Annulus distance (AD), which is defined as

AD(spectrum1, spectrum2) = |mean_radial(spectrum1) - mean_radial(spectrum2)| + |std_radial(spectrum1) - std_radial(spectrum2)|

Note that this is a quite course-grained metric, as it only captures the first and second momenta of the distributions in the radial direction. It neglects higher order momenta, details of angular distribution, and how radial and angular dependencies interact. In particular, it is not sensitive to higher order statistics, such as complex spacing ratio of the eigenvalues.

## Results and Discussion
