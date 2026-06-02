# FaMI4100-MLKem-PostQuantum

This repo aims to build the ML-KEM encryption (old name CRYSTALKyber, standardized name FIPS203). A lattice approach based on module learning with errors (M-LWE) problem, it is a key encapsulation mechanism (KEM) designed to be resistant to cryptanalytic attacks with future powerful quantum computers. This won NIST competition for the first post-quantum cryptography (PQC) standard.

Presentation file at: <https://colab.research.google.com/github/k1m1nonawa4869/FaMI4100-MLKem-PostQuantum/blob/main/presentation/ML-KEM.ipynb>

Pure code file is placed at "code" folder. It includes:

* ml_kem_core.py: the core file contain all needed algorithms of the ML-KEM
* alice_bob_demo.py: demostartion how to run a simulation in real life computer: generating key, encap, decaps,.
* benchmark.py: validate runtime of the algorithm, and comapre with similar algorithm like RSA/ECC
* math_demo.py: as ML-KEMS put heavily mathmatical topic, this file is purely to demo those object.
