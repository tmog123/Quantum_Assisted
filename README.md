# Quantum-Assisted

Code to implement:
Quantum Assisted Simulator - https://arxiv.org/pdf/2011.06911.pdf
Truncated Taylor Quantum Simulator - https://arxiv.org/pdf/2103.05500.pdf
Classical Quantum Fast Forwarding - https://arxiv.org/pdf/2104.01931.pdf

If you use our code, please cite our respective papers.

Refer to the Jupyter notebook tutorial to see how the code is meant to work for the quantum simulators (name is: Tutorial on how to use (Start here))


Things done: Created the stuff needed to do simple IQAE, TTQS, QAS, CQFF. Also
created the ability to interface with IBMQ quantum computers

Things to do: Create the code for more advanced plotting and analyzing

QCQP: NOT IMPLEMENTED YET. So far only optimization routine that has been implemented is eigh method (diagonalization). For TTQS/IQAE, should look into implementing a QCQP method in optimizers.py


Dependencies:

1. Qiskit
2. Numpy
3. Scipy
4. matplotlib

![image](https://user-images.githubusercontent.com/41319311/111591097-f44d8d00-8801-11eb-93a8-3804b6b6de88.png)

pauli_class_package:
1) Handles multiplication of pauli strings

Objects: Paulistring
Functions: pauli_combine, pauli_helper

hamiltonian_class_package:
1) Contains all the information of the Hamiltonian

REQUIRES PAULI CLASS

Objects: Hamiltonian
Functions: transverse_ising_model_1d, heisemberg_XYZ_model

ansatz_class_package:
1) Generates Ansatzes
2) Contains initial state class representation

Objects: Ansatz, moment, Initialstate
Functions: initial_ansatz, gen_next_ansatz

matrix_class_package:
1) Contains the representations for the unevaluated E and D matrices (each element is a list of paulistring objects that will be passed to backend to evaluate)

post_processing:

1. The routine to solve the generalised eigenvalue problem

optimizers:
1) The optimizers

plotting_package:
1) Does the plots
