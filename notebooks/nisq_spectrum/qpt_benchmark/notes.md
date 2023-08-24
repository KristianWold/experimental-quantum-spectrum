# Quantum Process Tomography Benchmark

## Setup

We perform benchmarks on the full QPT pipeline, starting by generating synthetic noisy data from ground truth SPAM models and diluted unitary models. 

### Synthetic SPAM

The ground truth SPAM models are modelled as diluted ideal SPAM, i.e., with $p=0.8$ for both inital state and POVM. The POVM part is full description. This provides a model for strongly corrupting SPAM errors, more so than in typical realistic hardware. This provides a worst case benchmark, highlighting the robustness of the procedure even when data is scarce and noisy.

The synthetic data is simulated with 1024 shots. All $6^n$ combinations of Pauli-strings is generated to fit the SPAM models. 

### Synthetic Quantum Channel

The ground truth Quantum Channel is modelled as a Diluted Unitary, where unitary part is (approximatly Haar) random and Kraus part is random with rank $r=d$. $p=0.5$. The idea is to provide a benchmark on a Quantum channel that is complicated and "interesting", with a high number of degrees of freedom, yet not completely random: The channel has a distinct and generic unitary part. Yet, it is highly dissipative. This unitary part is coated in a high rank dissipative channel, although not full rank.

For the quantum maps, $N_{map} = 1784$ and $N_{map} = 8704$ Pauli-strings are generated and fitted on. These are matching parameters corresponding to the experiments performed on the IBM platform, and should provide good benchmark showing the legitamacy of our experimental results.

### SPAM and Quantum Channel Models

The SPAM models fitted to the synthetic data have learn a imperfect inital state $\rho_0$, along with a Corruption Matrix as the POVM. This choice is motivated by results indicating that Corruption Matrix is favorable, over the full POVM description, when fitting on scarse and noisy data. At $1024$ shots and $6^n$ Pauli-strings, there is simply not enough data to reliably learn a full POVM description, resulting in a overall worse-performing model compared to Corruption Matrix.

## Results

Here we present and discuss the results of the QPT Benchmarks

### R2 Training loss and Channel Fidelity Loss
During training of the Quantum Channel Model, we monitor the R2 training loss over the probabilities resulting from Pauli-string measurement values. This is a measure on how well the model is able to replicate the statisics of the observed data resulting from the (syntheic) experiment. Since this is a synthetic experiemnt, we have access to the ground truth quantum channel generating the data. We evaluate the quality of the model by calculating the quantum channel fidelity against the ground truth channel. This measure the quality of the model in the most absolute sense: QCF = 1 if and only if the model is mathematically identical to the ground truth. Any deviation between them will result in $QCF<1$. 

### Stability of Retrived Quantum Spectra

The retieved models deviated substantially from the ground truth in terms of quantum state fidelity, meaning there are significant details that the model fails to pick up during training. However, it is not obvious that these details are important for capturing relevant details of specific aspects of the quantum channel. For example, relevant for us, can one accuratly recover the spectrum of the quantum channel? The results suggest yes. The spectrum is largly recovered 

