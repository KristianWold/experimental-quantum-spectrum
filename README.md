# Spectra of noisy parameterized quantum circuits: Single-Ring universality

This repository contains the source code, data, and analysis scripts for the paper [Spectra of noisy parameterized quantum circuits: Single-Ring universality](https://arxiv.org/pdf/2405.11625), which explores the retrival and analysis of quantum processes implemented as quantum circuits on real quantum hardware. In particular, the spectra of the quantum circuits are analysed, and compared to known theory from Random Matrix Theory.

## Repository Structure
```
├── analysis/                                       # subfolders containing notebooks, data, and resulting figures
│   ├── atypical_quantum_maps/  
│   │   ├── data/                                   # raw experimental and synthetic data
│   │   ├── figures/                                # figures resulting from notebooks
│   │   ├── Atypical_Quantum_Maps_Analysis.ipynb    # notebooks for analysis and visualisation
│   │   └── ...
│   ├── concatenation/
│   └── ...
├── src_tf/                                         # source code                 
│   ├── README.md
│   ├── experimental.py                   
│   └── ...
├── LICENSE.md
└── ...
```

## Requirements

Main libraries used are
- qiskit
- tensorflow
- matplotlib
- seaborn
- jupyter

For an exhaustive list, see [requirements.txt](requirements.txt) for Windows dependencies.

## Contact

personal email: kristian.wold@hotmail.com
university email: krisw@oslomet.no


